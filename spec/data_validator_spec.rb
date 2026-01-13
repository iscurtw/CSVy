require 'rspec'
require_relative '../lib/data_validator'
require 'tempfile'
require 'csv'

RSpec.describe DataValidator do
  let(:validator) { DataValidator.new }
  let(:temp_file) { Tempfile.new(['test', '.csv']) }

  after(:each) do
    temp_file.close
    temp_file.unlink
  end

  describe '#validate' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age', 'city']) do |csv|
        csv << ['Alice', '30', 'New York']
        csv << ['Bob', '25', 'Los Angeles']
        csv << ['Alice', '30', 'New York'] # duplicate
        csv << ['', '', ''] # empty row
      end
    end

    it 'detects empty rows' do
      report = validator.validate(temp_file.path)
      empty_issue = report[:issues].find { |i| i.include?('empty rows') }
      expect(empty_issue).not_to be_nil
    end

    it 'detects duplicate rows' do
      report = validator.validate(temp_file.path)
      duplicate_issue = report[:issues].find { |i| i.include?('duplicate rows') }
      expect(duplicate_issue).not_to be_nil
    end

    it 'returns correct row count' do
      report = validator.validate(temp_file.path)
      expect(report[:total_rows]).to eq(4)
    end
  end

  describe '#statistics' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'score']) do |csv|
        csv << ['Alice', '10']
        csv << ['Bob', '20']
        csv << ['Charlie', '30']
      end
    end

    it 'generates numeric statistics' do
      stats = validator.statistics(temp_file.path)
      expect(stats['score']).to have_key(:mean)
      expect(stats['score']).to have_key(:min)
      expect(stats['score']).to have_key(:max)
    end

    it 'calculates correct mean' do
      stats = validator.statistics(temp_file.path)
      expect(stats['score'][:mean]).to eq(20.0)
    end
  end

  describe '#profile' do
    before do
      CSV.open(temp_file.path, 'w', write_headers: true, headers: ['name', 'age']) do |csv|
        csv << ['Alice', '30']
        csv << ['Bob', '25']
        csv << ['Charlie', '30']
      end
    end

    it 'generates column profiles' do
      profile = validator.profile(temp_file.path)
      expect(profile[:column_profiles]).to have_key('name')
      expect(profile[:column_profiles]).to have_key('age')
    end

    it 'detects unique counts' do
      profile = validator.profile(temp_file.path)
      expect(profile[:column_profiles]['name'][:unique_count]).to eq(3)
      expect(profile[:column_profiles]['age'][:unique_count]).to eq(2)
    end
  end
end
