
# Film Roll Photo Processor

## Installation

### Option 1: Run the Executable (Easiest)
1. Double-click `FilmRollProcessor.exe`
2. Windows may show a security warning - click "More info" then "Run anyway"
3. The application will start and open in your browser automatically

### Option 2: Use the Installer
1. Run `FilmRollProcessor_Setup.exe`
2. Follow the installation wizard
3. Launch from Start Menu or Desktop shortcut

## Usage

1. Click "Select Photos" to choose your scanned film images
2. Adjust processing settings in the right panel:
   - Auto White Balance: Corrects color casts
   - Auto Straighten: Aligns crooked scans
   - Remove Borders: Crops out scanner edges
3. Click "Process Photos" to start
4. Processed images will be saved to: `Documents\FilmRollProcessor\Output`

## Output

- Files are saved as 16-bit TIFF for maximum quality
- EXIF metadata is preserved from RAW files
- Photos are automatically grouped by film roll

## Supported Formats

- RAW: .ARW, .CR2, .NEF, .DNG, .RAF, .ORF, .RW2, .RAW
- Standard: .JPG, .PNG, .TIF, .TIFF

## Troubleshooting

### Application won't start
- Make sure you have Windows 10 or later
- Try running as Administrator
- Check Windows Defender isn't blocking it

### Processing is slow
- Large RAW files take time to process
- Close other applications to free up memory
- Be patient - 16-bit processing is CPU intensive

### EXIF data not preserved
- Some scanners don't embed proper EXIF
- Try using the original RAW files if available

## System Requirements

- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

## Contact

For support or issues, please contact: your-email@example.com
