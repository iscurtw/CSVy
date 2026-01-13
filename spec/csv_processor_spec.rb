require 'rspec'
require_relative '../lib/csv_processor'
require_relative '../lib/csv_cleaner'
require 'tempfile'
require 'csv'

RSpec.describe CSVProcessor do
  let(:temp_file) { Tempfile.new(['test', '.csv']) }
  let(:temp_file2) { Tempfile.new(['test2', '.csv']) }

  after(:each) do
    temp_file.close
    temp_file.unlink
    temp_file2.close
    temp_file2.unlink
  end

  describe '.clean' do
    context 'with a valid CSV file' do
      before do
        CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age', 'city']) do |csv|
          csv << ['Alice', '30', 'New York']
          csv << ['Bob', '25', 'Los Angeles']
          csv << ['Alice', '30', 'New York'] # duplicate
          csv << ['', '', ''] # empty row
          csv << ['Charlie', '35', 'Chicago']
        end
      end

      it 'returns true when cleaning succeeds' do
        expect(CSVProcessor.clean(temp_file.path)).to be true
      end

      it 'creates a cleaned output file' do
        CSVProcessor.clean(temp_file.path)
        output_file = temp_file.path.gsub('.csv', '_cleaned.csv')
        expect(File.exist?(output_file)).to be true
        File.delete(output_file) if File.exist?(output_file)
      end
    end

    context 'with a non-existent file' do
      it 'returns false' do
        expect(CSVProcessor.clean('nonexistent.csv')).to be false
      end
    end
  end

  describe '.merge' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
        csv << ['Alice', '30']
        csv << ['Bob', '25']
      end

      CSV.open(temp_file2.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
        csv << ['Charlie', '35']
        csv << ['David', '40']
      end
    end

    it 'returns true when merging succeeds' do
      output_file = 'test_merged.csv'
      expect(CSVProcessor.merge(temp_file.path, temp_file2.path, output_file)).to be true
      File.delete(output_file) if File.exist?(output_file)
    end

    it 'creates a merged output file' do
      output_file = 'test_merged.csv'
      CSVProcessor.merge(temp_file.path, temp_file2.path, output_file)
      expect(File.exist?(output_file)).to be true
      File.delete(output_file) if File.exist?(output_file)
    end

    context 'with non-existent files' do
      it 'returns false' do
        expect(CSVProcessor.merge('nonexistent1.csv', 'nonexistent2.csv')).to be false
      end
    end
  end
end
