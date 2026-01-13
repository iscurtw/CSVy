require 'csv'
require 'logger'

class CSVDiagnostics
  attr_reader :file_path, :data, :logger, :sample_size

  # Common missing value representations
  MISSING_INDICATORS = ['', ' ', 'NA', 'N/A', 'NULL', 'null', 'None', 'none', 
                        '-', '--', '---', 'NaN', 'nan', '#N/A', '?', 'missing', 
                        '-999', '-9999', '999', '9999'].freeze

  def initialize(file_path, sample_size: 10000)
    @file_path = file_path
    @sample_size = sample_size
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
    load_data
  end

  def load_data
    all_data = CSV.read(@file_path, headers: true)
    total_rows = all_data.length
    
    # Sample if file is large
    if total_rows > @sample_size
      @logger.warn "Large file detected (#{total_rows} rows). Sampling #{@sample_size} rows for analysis."
      indices = (0...total_rows).to_a.sample(@sample_size).sort
      @data = CSV::Table.new(indices.map { |i| all_data[i] })
    else
      @data = all_data
    end
    
    @logger.info "Loaded #{@data.length} rows from #{@file_path}"
  end

  def diagnose
    puts "\n" + "="*80
    puts "CSV DIAGNOSTIC REPORT"
    puts "="*80
    puts "File: #{@file_path}"
    puts "Rows analyzed: #{@data.length}"
    puts "Columns: #{@data.headers.length}"
    puts "="*80
    
    @data.headers.each_with_index do |column, idx|
      puts "\n#{idx + 1}. Column: '#{column}'"
      puts "-" * 80
      
      diagnose_column(column)
    end
    
    puts "\n" + "="*80
    puts "OVERALL RECOMMENDATIONS"
    puts "="*80
    generate_recommendations
    puts "="*80
  end

  def diagnose_column(column_name)
    values = @data.map { |row| row[column_name] }
    non_empty = values.reject { |v| v.nil? || v.to_s.strip.empty? }
    
    # Basic stats
    puts "Total values: #{values.length}"
    puts "Empty/nil values: #{values.length - non_empty.length} (#{percent(values.length - non_empty.length, values.length)}%)"
    
    # Detect missing value indicators
    missing_types = detect_missing_indicators(values)
    if missing_types.any?
      puts "Missing value indicators detected: #{missing_types.keys.join(', ')}"
      missing_types.each { |indicator, count| puts "  '#{indicator}': #{count} occurrences" }
    end
    
    # Data type analysis
    type_analysis = analyze_types(non_empty)
    puts "\nData type distribution:"
    type_analysis.each { |type, pct| puts "  #{type}: #{pct}%" }
    
    if type_analysis.keys.length > 1
      puts "⚠ WARNING: Mixed data types detected!"
      puts "  Recommendation: Review column for data entry errors or split into multiple columns"
    end
    
    # Numeric column analysis
    if type_analysis['numeric'] && type_analysis['numeric'] > 50
      analyze_numeric_column(column_name, non_empty)
    end
    
    # Text column analysis
    if type_analysis['text'] && type_analysis['text'] > 50
      analyze_text_column(column_name, non_empty)
    end
    
    # Uniqueness
    unique_count = non_empty.uniq.length
    puts "\nUnique values: #{unique_count} (#{percent(unique_count, non_empty.length)}% cardinality)"
    
    if unique_count <= 10
      puts "Sample values: #{non_empty.uniq.first(10).join(', ')}"
    elsif unique_count == non_empty.length
      puts "⚠ All values are unique - potential identifier column"
    end
  end

  def detect_missing_indicators(values)
    found = {}
    MISSING_INDICATORS.each do |indicator|
      count = values.count { |v| v.to_s.strip == indicator }
      found[indicator] = count if count > 0
    end
    found
  end

  def analyze_types(values)
    total = values.length.to_f
    return {} if total.zero?
    
    numeric = 0
    date_like = 0
    text = 0
    
    values.each do |v|
      str = v.to_s.strip
      if numeric?(str)
        numeric += 1
      elsif date_like?(str)
        date_like += 1
      else
        text += 1
      end
    end
    
    result = {}
    result['numeric'] = ((numeric / total) * 100).round(1) if numeric > 0
    result['date'] = ((date_like / total) * 100).round(1) if date_like > 0
    result['text'] = ((text / total) * 100).round(1) if text > 0
    result
  end

  def analyze_numeric_column(column_name, values)
    numeric_values = values.map { |v| extract_number(v) }.compact
    return if numeric_values.empty?
    
    sorted = numeric_values.sort
    min = sorted.first
    max = sorted.last
    mean = numeric_values.sum / numeric_values.size.to_f
    median = sorted[sorted.size / 2]
    
    puts "\nNumeric statistics:"
    puts "  Min: #{min}"
    puts "  Max: #{max}"
    puts "  Mean: #{mean.round(2)}"
    puts "  Median: #{median}"
    puts "  Range: #{max - min}"
    
    # Check for sentinel values
    suspicious = []
    suspicious << "-999 or -9999" if sorted.any? { |v| v <= -999 }
    suspicious << "999 or 9999" if sorted.any? { |v| v >= 999 }
    suspicious << "All zeros" if sorted.all?(&:zero?)
    
    if suspicious.any?
      puts "⚠ Potential sentinel/missing values detected: #{suspicious.join(', ')}"
      puts "  Recommendation: Verify if these are real values or missing data indicators"
    end
    
    # Outlier detection
    detect_outliers(column_name, sorted)
    
    # Distribution check
    check_distribution(sorted)
  end

  def detect_outliers(column_name, sorted_values)
    return if sorted_values.size < 4
    
    q1 = sorted_values[sorted_values.size / 4]
    q3 = sorted_values[(sorted_values.size * 3) / 4]
    iqr = q3 - q1
    
    return if iqr.zero?
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = sorted_values.select { |v| v < lower_bound || v > upper_bound }
    
    if outliers.any?
      puts "\nOutlier detection (IQR method):"
      puts "  Q1: #{q1}, Q3: #{q3}, IQR: #{iqr.round(2)}"
      puts "  Bounds: [#{lower_bound.round(2)}, #{upper_bound.round(2)}]"
      puts "  Outliers found: #{outliers.length} (#{percent(outliers.length, sorted_values.length)}%)"
      puts "  Outlier values: #{outliers.uniq.first(10).join(', ')}"
      
      if outliers.length > sorted_values.length * 0.1
        puts "⚠ >10% outliers detected - IQR method may be too aggressive for this data"
      end
    end
  end

  def check_distribution(sorted_values)
    # Simple distribution check
    median = sorted_values[sorted_values.size / 2]
    p10 = sorted_values[sorted_values.size / 10]
    p90 = sorted_values[(sorted_values.size * 9) / 10]
    
    puts "\nDistribution:"
    puts "  10th percentile: #{p10}"
    puts "  50th percentile (median): #{median}"
    puts "  90th percentile: #{p90}"
    
    # Check for skewness
    range_lower = median - p10
    range_upper = p90 - median
    
    if range_upper > range_lower * 3
      puts "  Distribution: Right-skewed (long tail of high values)"
    elsif range_lower > range_upper * 3
      puts "  Distribution: Left-skewed (long tail of low values)"
    else
      puts "  Distribution: Roughly symmetric"
    end
  end

  def analyze_text_column(column_name, values)
    # Check for leading/trailing whitespace issues
    whitespace_issues = values.count { |v| v.to_s != v.to_s.strip }
    if whitespace_issues > 0
      puts "⚠ #{whitespace_issues} values have leading/trailing whitespace"
    end
    
    # Check for case inconsistencies
    downcased = values.map { |v| v.to_s.downcase }
    if downcased.uniq.length < values.uniq.length
      puts "⚠ Case inconsistencies detected (e.g., 'Toronto' vs 'toronto')"
    end
    
    # Length analysis
    lengths = values.map { |v| v.to_s.length }
    puts "\nText length:"
    puts "  Min: #{lengths.min}, Max: #{lengths.max}, Avg: #{(lengths.sum / lengths.size.to_f).round(1)}"
  end

  def generate_recommendations
    issues = []
    
    @data.headers.each do |column|
      values = @data.map { |row| row[column] }
      non_empty = values.reject { |v| v.nil? || v.to_s.strip.empty? }
      
      # High missing rate
      missing_pct = ((values.length - non_empty.length) / values.length.to_f) * 100
      if missing_pct > 20
        issues << "Column '#{column}': #{missing_pct.round(1)}% missing data - consider imputation or removal"
      end
      
      # Mixed types
      type_analysis = analyze_types(non_empty)
      if type_analysis.keys.length > 1
        issues << "Column '#{column}': Mixed data types - needs manual review"
      end
      
      # All unique (potential ID)
      if non_empty.uniq.length == non_empty.length && non_empty.length > 10
        issues << "Column '#{column}': Appears to be an identifier - safe to exclude from analysis"
      end
    end
    
    if issues.empty?
      puts "✓ No major issues detected. Data quality looks good!"
    else
      puts "Issues found:"
      issues.each_with_index { |issue, i| puts "#{i + 1}. #{issue}" }
    end
    
    puts "\nNext steps:"
    puts "1. Run 'validate' for detailed statistics"
    puts "2. Run 'clean' to see proposed transformations (side-by-side comparison)"
    puts "3. Import cleaned data into DeepNote for modeling"
  end

  private

  def numeric?(str)
    return false if str.nil? || str.empty?
    # Check if it's a pure number or has number with common suffixes
    str.match?(/^-?\d+\.?\d*$/) || str.match?(/^-?\d+\.?\d*[eE][+-]?\d+$/)
  end

  def date_like?(str)
    return false if str.nil? || str.empty?
    # Simple date pattern check
    str.match?(/^\d{4}-\d{2}-\d{2}/) || str.match?(/^\d{2}\/\d{2}\/\d{4}/) || str.match?(/^\d{2}-\d{2}-\d{4}/)
  end

  def extract_number(str)
    # Try to extract number even if there's text
    match = str.to_s.match(/-?\d+\.?\d*/)
    match ? match[0].to_f : nil
  end

  def percent(part, total)
    return 0 if total.zero?
    ((part / total.to_f) * 100).round(1)
  end
end
