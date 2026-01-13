require 'csv'
require 'logger'

class DataValidator
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  # Validate data quality
  def validate(file_path)
    logger.info "Validating data from: #{file_path}"
    
    data = CSV.read(file_path, headers: true)
    
    report = {
      file: file_path,
      total_rows: data.length,
      total_columns: data.headers.length,
      columns: data.headers,
      issues: []
    }
    
    # Check for empty rows
    empty_rows = count_empty_rows(data)
    report[:issues] << "Found #{empty_rows} empty rows" if empty_rows > 0
    
    # Check for duplicates
    duplicate_rows = count_duplicates(data)
    report[:issues] << "Found #{duplicate_rows} duplicate rows" if duplicate_rows > 0
    
    # Check for missing values per column
    missing_values = check_missing_values(data)
    missing_values.each do |col, count|
      report[:issues] << "Column '#{col}' has #{count} missing values" if count > 0
    end
    
    # Check data types
    report[:column_types] = infer_types(data)
    
    # Check for potential issues
    report[:warnings] = check_warnings(data)
    
    logger.info "Validation complete. Found #{report[:issues].length} issues"
    report
  end

  # Generate statistics for a dataset
  def statistics(file_path)
    logger.info "Generating statistics for: #{file_path}"
    
    data = CSV.read(file_path, headers: true)
    stats = {}
    
    data.headers.each do |column|
      values = data.map { |row| row[column] }.compact
      
      if numeric_column?(values)
        stats[column] = numeric_stats(values.map(&:to_f))
      else
        stats[column] = categorical_stats(values)
      end
    end
    
    logger.info "Statistics generated for #{stats.keys.length} columns"
    stats
  end

  # Check data integrity
  def check_integrity(file_path, rules = {})
    logger.info "Checking data integrity against rules"
    
    data = CSV.read(file_path, headers: true)
    violations = []
    
    rules.each do |column, rule|
      data.each_with_index do |row, idx|
        value = row[column]
        
        case rule[:type]
        when :not_null
          violations << "Row #{idx + 1}: #{column} is null" if value.nil? || value.strip.empty?
        when :unique
          # Check if value appears more than once
          count = data.count { |r| r[column] == value }
          violations << "Row #{idx + 1}: #{column} value '#{value}' is not unique" if count > 1
        when :range
          val = value.to_f
          if val < rule[:min] || val > rule[:max]
            violations << "Row #{idx + 1}: #{column} value #{val} is out of range [#{rule[:min]}, #{rule[:max]}]"
          end
        when :pattern
          unless value.match?(rule[:regex])
            violations << "Row #{idx + 1}: #{column} value '#{value}' doesn't match pattern"
          end
        when :enum
          unless rule[:values].include?(value)
            violations << "Row #{idx + 1}: #{column} value '#{value}' is not in allowed values: #{rule[:values].join(', ')}"
          end
        end
      end
    end
    
    logger.info "Integrity check complete. Found #{violations.length} violations"
    violations
  end

  # Profile dataset
  def profile(file_path)
    logger.info "Profiling dataset: #{file_path}"
    
    data = CSV.read(file_path, headers: true)
    
    profile = {
      file: file_path,
      rows: data.length,
      columns: data.headers.length,
      memory_estimate: estimate_memory(data),
      column_profiles: {}
    }
    
    data.headers.each do |column|
      values = data.map { |row| row[column] }
      non_null_values = values.compact.reject { |v| v.to_s.strip.empty? }
      
      profile[:column_profiles][column] = {
        type: infer_type(non_null_values),
        missing_count: values.length - non_null_values.length,
        missing_percentage: ((values.length - non_null_values.length) / values.length.to_f * 100).round(2),
        unique_count: non_null_values.uniq.length,
        cardinality: (non_null_values.uniq.length / non_null_values.length.to_f * 100).round(2)
      }
      
      if numeric_column?(non_null_values)
        numeric_vals = non_null_values.map(&:to_f)
        profile[:column_profiles][column].merge!(
          min: numeric_vals.min,
          max: numeric_vals.max,
          mean: (numeric_vals.sum / numeric_vals.length.to_f).round(3),
          median: percentile(numeric_vals.sort, 50).round(3)
        )
      else
        top_values = non_null_values.each_with_object(Hash.new(0)) { |v, h| h[v] += 1 }
                                    .sort_by { |_, count| -count }
                                    .first(5)
                                    .to_h
        profile[:column_profiles][column][:top_values] = top_values
      end
    end
    
    logger.info "Profiling complete"
    profile
  end

  private

  def count_empty_rows(data)
    data.count do |row|
      row.fields.all? { |f| f.nil? || f.to_s.strip.empty? }
    end
  end

  def count_duplicates(data)
    seen = Set.new
    data.count do |row|
      key = row.fields.join('|')
      !seen.add?(key)
    end
  end

  def check_missing_values(data)
    missing = {}
    data.headers.each do |column|
      missing[column] = data.count { |row| row[column].nil? || row[column].to_s.strip.empty? }
    end
    missing
  end

  def infer_types(data)
    types = {}
    data.headers.each do |column|
      values = data.map { |row| row[column] }.compact.reject { |v| v.to_s.strip.empty? }
      types[column] = infer_type(values)
    end
    types
  end

  def infer_type(values)
    return :unknown if values.empty?
    
    sample = values.first(100)
    
    if sample.all? { |v| v.match?(/^\d+$/) }
      :integer
    elsif sample.all? { |v| v.match?(/^-?\d+\.?\d*$/) }
      :float
    elsif sample.all? { |v| v.match?(/^\d{4}-\d{2}-\d{2}/) }
      :date
    elsif sample.all? { |v| v.match?(/^(true|false)$/i) }
      :boolean
    else
      :string
    end
  end

  def numeric_column?(values)
    return false if values.empty?
    values.sample(100).all? { |v| v.to_s.match?(/^-?\d+\.?\d*$/) }
  end

  def numeric_stats(values)
    sorted = values.sort
    {
      count: values.length,
      min: sorted.first,
      max: sorted.last,
      mean: (values.sum / values.length.to_f).round(3),
      median: percentile(sorted, 50).round(3),
      q1: percentile(sorted, 25).round(3),
      q3: percentile(sorted, 75).round(3),
      std_dev: standard_deviation(values).round(3)
    }
  end

  def categorical_stats(values)
    frequency = values.each_with_object(Hash.new(0)) { |v, h| h[v] += 1 }
    {
      count: values.length,
      unique: values.uniq.length,
      mode: frequency.max_by { |_, count| count }[0],
      top_5: frequency.sort_by { |_, count| -count }.first(5).to_h
    }
  end

  def check_warnings(data)
    warnings = []
    
    # Check for columns with high missing rate
    data.headers.each do |column|
      missing_rate = data.count { |row| row[column].nil? || row[column].to_s.strip.empty? } / data.length.to_f
      warnings << "Column '#{column}' has #{(missing_rate * 100).round(1)}% missing values" if missing_rate > 0.5
    end
    
    # Check for low cardinality columns
    data.headers.each do |column|
      unique = data.map { |row| row[column] }.uniq.length
      cardinality = unique / data.length.to_f
      warnings << "Column '#{column}' has very low cardinality (#{unique} unique values)" if cardinality < 0.01 && data.length > 100
    end
    
    warnings
  end

  def estimate_memory(data)
    size_bytes = data.to_s.bytesize
    if size_bytes < 1024
      "#{size_bytes} bytes"
    elsif size_bytes < 1024 * 1024
      "#{(size_bytes / 1024.0).round(2)} KB"
    else
      "#{(size_bytes / (1024.0 * 1024)).round(2)} MB"
    end
  end

  def percentile(sorted_array, percentile)
    return 0 if sorted_array.empty?
    index = (percentile / 100.0) * (sorted_array.length - 1)
    lower = sorted_array[index.floor]
    upper = sorted_array[index.ceil]
    lower + (upper - lower) * (index - index.floor)
  end

  def standard_deviation(values)
    mean = values.sum / values.length.to_f
    variance = values.map { |v| (v - mean)**2 }.sum / values.length.to_f
    Math.sqrt(variance)
  end
end
