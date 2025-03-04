# HugeImage

HugeImage is a Python application for managing and organizing images with face detection capabilities, integrated with contact management from VCF files.

## Features

- Face detection in images
- Contact management from VCF files
- YAML-based metadata storage
- UUID-based person identification
- Automatic face-to-contact linking

## Requirements

- Python 3.x
- OpenCV (cv2)
- numpy
- PyYAML
- vobject (for VCF processing)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hugeImage.git
cd hugeImage
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install opencv-python numpy pyyaml vobject
```

## Project Structure

```
hugeImage/
├── hugeImage.py         # Main application file
├── assets/             # Image storage (gitignored)
│   ├── Image_Data/    # YAML metadata for images
│   └── People/        # YAML files for contacts
├── .gitignore
└── README.md
```

## Usage

1. Initialize the application:

```python
from hugeImage import HugeImage
app = HugeImage()
```

2. Process VCF contacts:

```python
app.process_vcf_contacts('path/to/contacts.vcf')
```

3. Update image metadata:

```python
app.update_image_metadata('image_id', 'person_uuid', face_index)
```

## Data Structure

### Person YAML

```yaml
contact:
  email: john.doe@email.com
  firstname: John
  lastname: Doe
  phone: +1234567890
created_at: '2024-03-14T12:00:00'
face_folders: []
uuid: ce6e0f78-2df9-4cf7-8797-8350b99a6ce4
```

### Image Metadata YAML

```yaml
image:
  description: Image description
  faces:
    face1:
      bottomright: [x, y]
      person: UUID
      size: WxH
      topleft: [x, y]
  id: image_id
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for face detection
- PyYAML for YAML processing
- vobject for VCF handling

## Contact

