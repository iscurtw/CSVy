require 'csv'
require 'logger'
require_relative 'csv_cleaner'
require_relative 'csv_merger'

class CSVProcessor
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  def self.clean(file_path)
    processor = new
    processor.clean(file_path)
  end

  def self.merge(file1, file2, output_file = 'merged.csv')
    processor = new
    processor.merge(file1, file2, output_file)
  end

  def clean(file_path)
    logger.info "Starting to clean file: #{file_path}"
    
    unless File.exist?(file_path)
      logger.error "File not found: #{file_path}"
      return false
    end

    begin
      cleaner = CSVCleaner.new(file_path)
      cleaned_data = cleaner.clean_data
      
      output_file = file_path.gsub('.csv', '_cleaned.csv')
      cleaner.save_to_csv(cleaned_data, output_file)
      
      logger.info "Successfully cleaned file. Output saved to: #{output_file}"
      true
    rescue StandardError => e
      logger.error "Error cleaning file: #{e.message}"
      false
    end
  end

  def merge(file1, file2, output_file = 'merged.csv')
    logger.info "Starting to merge files: #{file1} and #{file2}"
    
    unless File.exist?(file1) && File.exist?(file2)
      logger.error "One or both files not found"
      return false
    end

    begin
      merger = CSVMerger.new
      merged_data = merger.merge_files(file1, file2)
      merger.save_to_csv(merged_data, output_file)
      
      logger.info "Successfully merged files. Output saved to: #{output_file}"
      true
    rescue StandardError => e
      logger.error "Error merging files: #{e.message}"
      false
    end
  end

  def self.transform(file_path, options = {})
    processor = new
    processor.transform(file_path, options)
  end

  def transform(file_path, options = {})
    logger.info "Starting to transform file: #{file_path}"
    
    unless File.exist?(file_path)
      logger.error "File not found: #{file_path}"
      return false
    end

    begin
      # Custom transformation logic can be added here
      logger.info "Transformation completed"
      true
    rescue StandardError => e
      logger.error "Error transforming file: #{e.message}"
      false
    end
  end
end
