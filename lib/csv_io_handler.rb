require 'csv'
require 'logger'

class CSVIOHandler
  attr_reader :logger

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.level = Logger::INFO
  end

  # Read CSV from string
  def self.from_string(csv_string, headers: true)
    CSV.parse(csv_string, headers: headers)
  end

  # Read CSV from IO object
  def self.from_io(io_object, headers: true)
    CSV.parse(io_object.read, headers: headers)
  end

  # Write CSV to string
  def self.to_string(data)
    CSV.generate do |csv|
      csv << data.headers
      data.each { |row| csv << row }
    end
  end

  # Write CSV to IO object
  def self.to_io(data, io_object)
    CSV(io_object) do |csv|
      csv << data.headers
      data.each { |row| csv << row }
    end
  end

  # Read from clipboard (Windows)
  def self.from_clipboard
    require 'win32/clipboard' if RUBY_PLATFORM =~ /win32|mingw|mswin/
    
    if RUBY_PLATFORM =~ /win32|mingw|mswin/
      clipboard_data = Win32::Clipboard.data
      from_string(clipboard_data)
    else
      # Use pbpaste on macOS or xclip on Linux
      clipboard_data = `pbpaste 2>/dev/null || xclip -selection clipboard -o 2>/dev/null`
      from_string(clipboard_data)
    end
  end

  # Write to clipboard
  def self.to_clipboard(data)
    csv_string = to_string(data)
    
    if RUBY_PLATFORM =~ /win32|mingw|mswin/
      require 'win32/clipboard'
      Win32::Clipboard.set_data(csv_string)
    else
      IO.popen('pbcopy', 'w') { |io| io.write(csv_string) } rescue nil
      IO.popen('xclip -selection clipboard', 'w') { |io| io.write(csv_string) } rescue nil
    end
    
    csv_string
  end

  # Read from STDIN
  def self.from_stdin
    from_io(STDIN)
  end

  # Write to STDOUT
  def self.to_stdout(data)
    to_io(data, STDOUT)
  end

  # Detect source type and read accordingly
  def self.read_smart(source)
    case source
    when String
      if File.exist?(source)
        CSV.read(source, headers: true)
      else
        from_string(source)
      end
    when IO, StringIO
      from_io(source)
    when :clipboard
      from_clipboard
    when :stdin
      from_stdin
    else
      raise ArgumentError, "Unknown source type: #{source.class}"
    end
  end

  # Detect destination and write accordingly
  def self.write_smart(data, destination)
    case destination
    when String
      CSV.open(destination, 'w', write_headers: true, headers: data.headers) do |csv|
        data.each { |row| csv << row }
      end
    when IO, StringIO
      to_io(data, destination)
    when :clipboard
      to_clipboard(data)
    when :stdout
      to_stdout(data)
    else
      raise ArgumentError, "Unknown destination type: #{destination.class}"
    end
  end
end
