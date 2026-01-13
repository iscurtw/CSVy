require 'rspec'
require_relative '../lib/csv_merger'
require 'tempfile'
require 'csv'

RSpec.describe CSVMerger do
  let(:temp_file1) { Tempfile.new(['test1', '.csv']) }
  let(:temp_file2) { Tempfile.new(['test2', '.csv']) }
  let(:merger) { CSVMerger.new }

  after(:each) do
    temp_file1.close
    temp_file1.unlink
    temp_file2.close
    temp_file2.unlink
  end

  describe '#concatenate' do
    before do
      CSV.open(temp_file1.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
        csv << ['Alice', '30']
        csv << ['Bob', '25']
      end

      CSV.open(temp_file2.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
        csv << ['Charlie', '35']
        csv << ['David', '40']
      end
    end

    it 'concatenates two CSV files' do
      result = merger.concatenate(temp_file1.path, temp_file2.path)
      expect(result.length).to eq(4)
    end

    it 'preserves all data from both files' do
      result = merger.concatenate(temp_file1.path, temp_file2.path)
      names = result.map { |row| row['name'] }
      expect(names).to contain_exactly('Alice', 'Bob', 'Charlie', 'David')
    end
  end

  describe '#join_on_column' do
    before do
      CSV.open(temp_file1.path, 'w', write_headers: true, headers: ['id', 'name']) do |csv|
        csv << ['1', 'Alice']
        csv << ['2', 'Bob']
      end

      CSV.open(temp_file2.path, 'w', write_headers: true, headers: ['id', 'city']) do |csv|
        csv << ['1', 'New York']
        csv << ['2', 'Los Angeles']
      end
    end

    it 'joins two files on specified column' do
      result = merger.join_on_column(temp_file1.path, temp_file2.path, key_column: 'id')
      expect(result.length).to eq(2)
    end

    it 'combines columns from both files' do
      result = merger.join_on_column(temp_file1.path, temp_file2.path, key_column: 'id')
      expect(result.headers).to include('id', 'name', 'city')
    end
  end

  describe '#save_to_csv' do
    before do
      CSV.open(temp_file1.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
        csv << ['Alice', '30']
      end
    end

    it 'saves CSV data to file' do
      data = CSV.read(temp_file1.path, headers: true)
      output_file = 'test_output.csv'
      
      merger.save_to_csv(data, output_file)
      
      expect(File.exist?(output_file)).to be true
      File.delete(output_file) if File.exist?(output_file)
    end
  end
end
