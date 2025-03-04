import os
import shortuuid
import yaml
import cv2
from pathlib import Path
import requests
import base64
import platform
import subprocess
import numpy as np
from PIL import Image
from skimage import exposure
import uuid
import datetime
import sys
from tabulate import tabulate

class ImageProcessor:
    def __init__(self, base_path="hugeImage/assets"):
        """Initialize the ImageProcessor with required packages"""
        # Check and install required packages
        try:
            from tabulate import tabulate
        except ImportError:
            print("Installing required package: tabulate")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        
        # Setup directory structure
        self.base_path = Path(base_path)
        self.directories = {
            'images': self.base_path / 'Images',
            'thumbnails': self.base_path / 'Thumbnails',
            'faces': self.base_path / 'Faces',
            'image_data': self.base_path / 'Image_Data',
            'people': self.base_path / 'People'
        }
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path):
        """Main function to process an image through all steps"""
        # Generate unique ID
        unique_id = self._generate_unique_id()
        
        # Process and save original image
        image_ext = Path(image_path).suffix
        
        new_image_path = self._save_original_image(image_path, unique_id, image_ext)
        
        # Create and save thumbnail
        self._create_thumbnail(new_image_path, unique_id)
        
        # Process faces
        faces_data = self._process_faces(new_image_path, unique_id)
        
        # Generate image description using Ollama
        description = self._generate_description(new_image_path)
        
        # Save metadata
        self._save_metadata(unique_id, description, faces_data)
        
        return unique_id

    def _generate_unique_id(self):
        """Generate a 12-character unique ID"""
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        return shortuuid.ShortUUID(alphabet=alphabet).random(length=12)

    def _save_original_image(self, image_path, unique_id, ext):
        """Save the original image with new UUID-based name and delete original"""
        new_path = self.directories['images'] / f"img_{unique_id}{ext}"
        
        # Copy the image to new location
        Image.open(image_path).save(new_path)
        
        # Delete the original file
        try:
            os.remove(image_path)
            print(f"Original file deleted: {image_path}")
        except OSError as e:
            print(f"Warning: Could not delete original file: {e}")
        
        return new_path

    def _create_thumbnail(self, image_path, unique_id):
        """Create and save 500x500 thumbnail using OpenCV"""
        # Read image
        image = cv2.imread(str(image_path))
        
        # Calculate new dimensions while maintaining aspect ratio
        height, width = image.shape[:2]
        max_size = 500
        
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size/height))
        else:
            new_width = max_size
            new_height = int(height * (max_size/width))
        
        # Resize image
        thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save thumbnail
        thumb_path = self.directories['thumbnails'] / f"thumb_{unique_id}.jpg"
        cv2.imwrite(str(thumb_path), thumbnail)

    def _process_faces(self, image_path, unique_id):
        """Detect and save faces using OpenCV instead of face_recognition"""
        # Load the pre-trained face detection classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load image using OpenCV
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with different parameters for better accuracy
        face_locations = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Found {len(face_locations)} faces in the image")
        
        faces_data = {}
        for idx, (x, y, w, h) in enumerate(face_locations, 1):
            # Calculate face size
            face_width = w
            face_height = h
            print(f"Face {idx} size: {face_width}x{face_height} pixels")
            
            # Add margin around face
            margin = 30  # pixels
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2*margin, image.shape[1] - x)
            h = min(h + 2*margin, image.shape[0] - y)
            
            # Extract the face region
            face_img = image[y:y+h, x:x+w]
            
            # Save face to Faces folder
            face_path = self.directories['faces'] / f"img_{unique_id}_face{idx}.jpg"
            cv2.imwrite(str(face_path), face_img)
            print(f"Saved face {idx} to {face_path}")
            
            # Store face location data
            faces_data[f'face{idx}'] = {
                'person': 'UNKNOWN',
                'topleft': [x, y],
                'bottomright': [x+w, y+h],
                'size': f"{face_width}x{face_height}"
            }
        
        if not faces_data:
            print("Warning: No faces were detected. Trying with different parameters...")
            
            # Try again with more lenient parameters
            face_locations = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            for idx, (x, y, w, h) in enumerate(face_locations, 1):
                face_width = w
                face_height = h
                print(f"Face {idx} size: {face_width}x{face_height} pixels")
                
                margin = 30
                x = max(x - margin, 0)
                y = max(y - margin, 0)
                w = min(w + 2*margin, image.shape[1] - x)
                h = min(h + 2*margin, image.shape[0] - y)
                
                face_img = image[y:y+h, x:x+w]
                face_path = self.directories['faces'] / f"img_{unique_id}_face{idx}.jpg"
                cv2.imwrite(str(face_path), face_img)
                
                faces_data[f'face{idx}'] = {
                    'person': 'UNKNOWN',
                    'topleft': [x, y],
                    'bottomright': [x+w, y+h],
                    'size': f"{face_width}x{face_height}"
                }
                print(f"Saved face {idx} to {face_path} (second pass)")
        
        if not faces_data:
            print("Warning: No faces were detected. This might be due to:")
            print("- Face angles (profile views are harder to detect)")
            print("- Image resolution or quality")
            print("- Lighting conditions")
            print("- Face size in the image")
        
        return faces_data

    def _generate_description(self, image_path):
        """Generate image description using Ollama"""
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare the prompt for Ollama
            prompt = "Describe this image in 2-3 concise sentences, focusing on the main elements and actions visible."
            
            # Make request to Ollama
            response = requests.post('http://localhost:11434/api/generate',
                json={
                    "model": "llava",  # or your preferred Ollama model with vision capabilities
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False
                })
            
            if response.status_code == 200:
                description = response.json()['response'].strip()
                print(f"Generated description: {description}")
                return description
            else:
                print(f"Error from Ollama API: {response.status_code}")
                return "Error generating description"
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Error generating description"

    def _save_metadata(self, unique_id, description, faces_data):
        """Save image metadata to YAML file"""
        # Convert tuple coordinates to lists for YAML compatibility
        faces_data_yaml = {}
        for face_key, face_info in faces_data.items():
            faces_data_yaml[face_key] = {
                'person': face_info['person'],
                'topleft': [int(x) for x in face_info['topleft']],  # Convert to int
                'bottomright': [int(x) for x in face_info['bottomright']],  # Convert to int
                'size': face_info['size']
            }
        
        metadata = {
            unique_id: {
                'description': description,
                'faces': faces_data_yaml
            }
        }
        
        yaml_path = self.directories['image_data'] / f"{unique_id}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(metadata, f, default_flow_style=False)

    def process_all_images(self):
        """Process all images following the exact flowchart requirements"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG')  # Added uppercase extensions
        input_dir = self.directories['images']
        
        print("\nStarting Image Processing Pipeline")
        print("=================================")
        print(f"Looking for images in: {input_dir}")
        
        # 1. Import Images - Modified to find all images
        input_images = []
        for ext in image_extensions:
            found_images = list(input_dir.glob(f'*{ext}'))
            # Only add images that don't start with 'img_' (to avoid processing already processed images)
            for img in found_images:
                if not img.name.startswith('img_'):
                    input_images.append(img)
        
        if not input_images:
            print("\nNo new images found!")
            print(f"Please add images to: {input_dir}")
            return
        
        print(f"\nFound {len(input_images)} new images to process:")
        for img in input_images:
            print(f"- {img.name}")
        
        # Process each image
        for img_path in input_images:
            print(f"\nProcessing: {img_path.name}")
            try:
                # 2. Create Unique Image ID
                unique_id = self._generate_unique_id()
                print(f"Generated UUID: {unique_id}")
                
                # 3. Rename and Save Image
                new_image_path = self._save_original_image(img_path, unique_id, img_path.suffix.lower())
                print(f"Saved image as: img_{unique_id}{img_path.suffix.lower()}")
                
                # 4. Create Thumbnail
                self._create_thumbnail(new_image_path, unique_id)
                print(f"Created thumbnail: thumb_{unique_id}.jpg")
                
                # 5. Generate Image Description
                description = self._generate_description(new_image_path)
                print("\nGenerated image description")
                
                # 6. Process Faces
                faces_data = self._process_faces(new_image_path, unique_id)
                
                if faces_data:
                    print(f"\nDetected {len(faces_data)} faces")
                    # Save metadata with UNKNOWN persons
                    self._save_metadata(unique_id, description, faces_data)
                    print("\nSaved metadata")
                    
                    # Show created files
                    print("\nCreated files:")
                    print(f"1. Image: {self.directories['images']}/img_{unique_id}{img_path.suffix.lower()}")
                    print(f"2. Thumbnail: {self.directories['thumbnails']}/thumb_{unique_id}.jpg")
                    print(f"3. Metadata: {self.directories['image_data']}/{unique_id}.yaml")
                    print(f"4. Face images in: {self.directories['faces']}")
                else:
                    print("No faces detected in this image")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        print("\nAll processing complete!")

    def show_image(self, image_path):
        """Display image using system default viewer"""
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', '-a', 'Preview', image_path])
            elif platform.system() == 'Linux':
                subprocess.run(['xdg-open', image_path])
            elif platform.system() == 'Windows':
                os.startfile(image_path)
            
            print(f"Opened image: {image_path}")
            print("Please look at the image and press Enter when ready...")
            input()
            
            # For macOS, close Preview after user presses Enter
            if platform.system() == 'Darwin':
                subprocess.run(['osascript', '-e', 'tell application "Preview" to quit'])
            
        except Exception as e:
            print(f"Error showing image: {e}")
            print(f"Image path: {image_path}")

    def sort_similar_faces(self):
        """Sort detected faces into groups of similar faces by moving them"""
        print("\nStarting face sorting...")
        
        # Get all faces from the Faces folder directly
        face_files = list(self.directories['faces'].glob('img_*.jpg'))
        
        if not face_files:
            print("No faces found to process")
            return
        
        print(f"Found {len(face_files)} faces to process")
        
        try:
            import face_recognition
            from PIL import Image
            import numpy as np
        except ImportError:
            print("Please install required packages:")
            print("pip install face-recognition Pillow numpy")
            return
        
        # Process all images
        known_faces = []
        non_face_images = []
        
        # First pass: Get face encodings from all images
        for face_file in face_files:
            try:
                img_id = face_file.stem.split('_')[1]
                image = face_recognition.load_image_file(str(face_file))
                face_locations = face_recognition.face_locations(image, model="hog")
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if len(face_encodings) > 0:
                    encoding = face_encodings[0]
                    known_faces.append({
                        'file': face_file,
                        'encoding': encoding,
                        'face_image': image,
                        'img_id': img_id,
                        'index': face_file.stem.split('_face')[-1]
                    })
                    print(f"Processed face: {face_file.name}")
                else:
                    non_face_images.append({
                        'file': face_file,
                        'face_image': image,
                        'img_id': img_id,
                        'index': face_file.stem.split('_face')[-1]
                    })
                    print(f"Found non-face image: {face_file.name}")
                    
            except Exception as e:
                print(f"Error processing {face_file}: {e}")
                continue
        
        # Second pass: Group similar faces
        person_groups = []
        tolerance = 0.6
        
        while known_faces:
            current_face = known_faces.pop(0)
            current_group = [current_face]
            
            i = 0
            while i < len(known_faces):
                # Get face distance
                distance = face_recognition.face_distance([current_face['encoding']], 
                                                        known_faces[i]['encoding'])[0]
                
                # More lenient matching criteria
                if distance <= tolerance:
                    if current_face['img_id'] != known_faces[i]['img_id']:
                        current_group.append(known_faces.pop(i))
                    else:
                        i += 1
                else:
                    # Check if it matches any other face in the group
                    matches_group = False
                    for grouped_face in current_group:
                        group_distance = face_recognition.face_distance([grouped_face['encoding']], 
                                                                      known_faces[i]['encoding'])[0]
                        if group_distance <= tolerance:
                            matches_group = True
                            break
                    
                    if matches_group and current_face['img_id'] != known_faces[i]['img_id']:
                        current_group.append(known_faces.pop(i))
                    else:
                        i += 1
            
            person_groups.append(current_group)
        
        # Save classified faces
        print("\nOrganizing faces into person folders...")
        processed_files = set()
        
        for idx, group in enumerate(person_groups, 1):
            person_dir = self.directories['faces'] / f"person{idx}"
            person_dir.mkdir(exist_ok=True)
            
            print(f"\nPerson {idx} has {len(group)} face(s):")
            for face_data in group:
                try:
                    # Move the face file to person directory
                    new_path = person_dir / face_data['file'].name
                    if not new_path.exists():
                        face_image_pil = Image.fromarray(face_data['face_image'])
                        face_image_pil.save(new_path, 'JPEG')
                        print(f"  Saved {face_data['file'].name} to {person_dir.name}")
                        processed_files.add(face_data['file'])
                    
                except Exception as e:
                    print(f"Error saving face {face_data['file'].name}: {e}")
                    continue
        
        # Save non-face images to dedicated folder
        if non_face_images:
            print("\nSaving non-face images...")
            non_face_dir = self.directories['faces'] / "non_face_images"
            non_face_dir.mkdir(exist_ok=True)
            
            for img_data in non_face_images:
                try:
                    new_path = non_face_dir / img_data['file'].name
                    if not new_path.exists():
                        face_image_pil = Image.fromarray(img_data['face_image'])
                        face_image_pil.save(new_path, 'JPEG')
                        print(f"Saved non-face image: {img_data['file'].name}")
                        processed_files.add(img_data['file'])
                except Exception as e:
                    print(f"Error saving non-face image {img_data['file'].name}: {e}")
                    continue
        
        # Delete processed files from faces folder
        print("\nCleaning up processed files...")
        for file_path in processed_files:
            try:
                file_path.unlink()
                print(f"Deleted processed file: {file_path.name}")
            except Exception as e:
                print(f"Error deleting {file_path.name}: {e}")
        
        print(f"\nImage sorting complete!")
        print(f"Organized faces into {len(person_groups)} person folders")
        if non_face_images:
            print(f"Saved {len(non_face_images)} non-face images to 'non_face_images' folder")
        
        # Cleanup empty folders
        self.cleanup_empty_folders()

    def cleanup_empty_folders(self):
        """Remove empty person folders"""
        print("\nCleaning up empty person folders...")
        
        # Get all person folders
        person_folders = [f for f in self.directories['faces'].glob('person*') if f.is_dir()]
        
        if not person_folders:
            print("No person folders found")
            return
        
        for person_dir in person_folders:
            try:
                # Get all face images in this folder
                face_files = list(person_dir.glob('*.jpg'))
                if len(face_files) <= 1:
                    person_dir.rmdir()
                    print(f"Removed empty person folder: {person_dir.name}")
            except Exception as e:
                print(f"Error processing folder {person_dir}: {e}")
                continue
        
        print("\nCleanup complete!")

    def create_or_update_vcf(self, contact_data=None):
        """Create or update VCF file with contacts"""
        vcf_path = Path('contacts.vcf')
        
        try:
            if contact_data:
                # Add new contact to VCF
                with open(vcf_path, 'a') as f:
                    f.write('BEGIN:VCARD\n')
                    f.write('VERSION:3.0\n')
                    f.write(f"N:{contact_data['lastname']};{contact_data['firstname']};;;\n")
                    f.write(f"FN:{contact_data['firstname']} {contact_data['lastname']}\n")
                    if 'phone' in contact_data:
                        f.write(f"TEL;TYPE=CELL:{contact_data['phone']}\n")
                    if 'email' in contact_data:
                        f.write(f"EMAIL:{contact_data['email']}\n")
                    f.write('END:VCARD\n\n')
                print(f"Added {contact_data['firstname']} {contact_data['lastname']} to contacts.vcf")
                return True
        except Exception as e:
            print(f"Error updating VCF: {e}")
            return False

    def read_vcf_contacts(self):
        """Read contacts from VCF file"""
        vcf_path = Path('contacts.vcf')  # Changed to root directory
        contacts = []
        
        try:
            if vcf_path.exists():
                current_contact = {}
                with open(vcf_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line == 'BEGIN:VCARD':
                            current_contact = {}
                        elif line.startswith('N:'):
                            parts = line[2:].split(';')
                            current_contact['lastname'] = parts[0]
                            current_contact['firstname'] = parts[1]
                        elif line.startswith('TEL;'):
                            current_contact['phone'] = line.split(':')[1]
                        elif line.startswith('EMAIL:'):
                            current_contact['email'] = line.split(':')[1]
                        elif line == 'END:VCARD':
                            if current_contact:
                                contacts.append(current_contact)
        except Exception as e:
            print(f"Error reading VCF: {e}")
        
        return contacts

    def create_new_contact(self):
        """Create a new contact interactively"""
        print("\nCreating new contact:")
        print("-" * 30)
        contact_data = {
            'firstname': input("First Name  : ").strip(),
            'lastname': input("Last Name   : ").strip(),
            'phone': input("Phone      : ").strip(),
            'email': input("Email      : ").strip()
        }
        
        # Confirm contact details
        print("\nContact Details:")
        print("-" * 30)
        for key, value in contact_data.items():
            print(f"{key.capitalize():<10}: {value or 'N/A'}")
        print("-" * 30)
        
        if input("\nSave contact? (y/n): ").lower() != 'y':
            print("Contact creation cancelled.")
            return None
        
        # Remove empty optional fields
        contact_data = {k: v for k, v in contact_data.items() if v}
        
        # Generate UUID for the contact
        person_uuid = str(uuid.uuid4())
        
        # Create YAML file
        yaml_path = self.directories['people'] / f"person_{person_uuid}.yaml"
        try:
            with open(yaml_path, 'w') as f:
                yaml.safe_dump({
                    'uuid': person_uuid,
                    'contact': contact_data,
                    'face_folders': [],
                    'created_at': datetime.datetime.now().isoformat()
                }, f, default_flow_style=False)
            
            # Update VCF file
            if self.create_or_update_vcf(contact_data):
                print(f"\nContact saved successfully to both YAML and VCF!")
            else:
                print(f"\nWarning: Contact saved to YAML but VCF update failed.")
            
            return person_uuid
        
        except Exception as e:
            print(f"Error creating contact: {e}")
            return None

    def display_terminal_contacts(self):
        """Display contacts in terminal for selection"""
        contacts = self.read_vcf_contacts()
        
        if not contacts:
            print("\nNo contacts available.")
            return None
        
        try:
            from tabulate import tabulate
        except ImportError:
            print("Installing tabulate package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
            from tabulate import tabulate
        
        while True:
            # Clear terminal
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Create table data
            headers = ["#", "First Name", "Last Name", "Phone", "Email"]
            table_data = []
            
            for idx, contact in enumerate(contacts, 1):
                table_data.append([
                    idx,
                    contact.get('firstname', 'N/A'),
                    contact.get('lastname', 'N/A'),
                    contact.get('phone', 'N/A'),
                    contact.get('email', 'N/A')
                ])
            
            # Display contacts table
            print("\nAvailable Contacts:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print("\nOptions:")
            print("- Enter number to select contact")
            print("- 'n' to create new contact")
            print("- 's' to skip")
            print("- 'q' to quit")
            
            choice = input("\nEnter choice: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'n':
                return 'new'
            elif choice == 's':
                return 'skip'
            else:
                try:
                    idx = int(choice)
                    if 1 <= idx <= len(contacts):
                        return contacts[idx-1]
                    else:
                        print("\nInvalid contact number. Press Enter to try again.")
                        input()
                except ValueError:
                    print("\nInvalid input. Press Enter to try again.")
                    input()

    def update_image_metadata(self, img_id, person_uuid, face_index):
        """Update image metadata YAML with person UUID"""
        try:
            yaml_path = self.directories['image_data'] / f"{img_id}.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    content = f.readlines()
                
                # Find and update the person field for the specific face
                in_target_face = False
                for i, line in enumerate(content):
                    if line.strip().startswith(f'face{face_index}:'):
                        in_target_face = True
                    elif in_target_face and line.strip().startswith('person:'):
                        # Update the person field while preserving indentation
                        indent = len(line) - len(line.lstrip())
                        content[i] = ' ' * indent + f'person: {person_uuid}\n'
                        break
                    elif in_target_face and (line.strip().startswith('face') or not line.strip()):
                        in_target_face = False
                
                # Write back the updated content
                with open(yaml_path, 'w') as f:
                    f.writelines(content)
                
                print(f"Updated metadata for face {face_index} in image {img_id}")
                return True
                    
        except Exception as e:
            print(f"Error updating metadata for image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def match_faces_to_contacts(self):
        """Match face folders to contacts"""
        print("\nMatching faces to contacts...")
        
        # Get top 5 person folders by size
        person_folders = []
        for folder in self.directories['faces'].glob('person*'):
            if folder.is_dir():
                num_faces = len(list(folder.glob('*.jpg')))
                person_folders.append((folder, num_faces))
        
        person_folders.sort(key=lambda x: x[1], reverse=True)
        top_folders = person_folders[:5]
        
        for folder, num_faces in top_folders:
            # Show first image from folder
            first_image = next(folder.glob('*.jpg'), None)
            if first_image:
                try:
                    # Display image
                    self.show_image(first_image)
                    
                    print(f"\nViewing person from {folder.name} ({num_faces} faces)")
                    
                    while True:
                        selected = self.display_terminal_contacts()
                        
                        if selected is None:  # Quit
                            return
                        elif selected == 'skip':
                            print("Skipping this person...")
                            break
                        elif selected == 'new':
                            # Create new contact and match
                            person_uuid = self.create_new_contact()
                            if person_uuid:
                                yaml_path = self.directories['people'] / f"person_{person_uuid}.yaml"
                                data = yaml.safe_load(open(yaml_path))
                                data['face_folders'] = [folder.name]
                                with open(yaml_path, 'w') as f:
                                    yaml.safe_dump(data, f, default_flow_style=False)
                                
                                # Update all face metadata in this folder
                                for face_file in folder.glob('*.jpg'):
                                    if face_file.name.startswith('img_'):
                                        img_id = face_file.stem.split('_')[1]
                                        face_index = face_file.stem.split('_face')[-1]
                                        self.update_image_metadata(img_id, person_uuid, face_index)
                                
                                print(f"\nSuccessfully created new contact and matched with {folder.name}")
                            break
                        else:
                            # Match with existing contact
                            person_uuid = str(uuid.uuid4())
                            yaml_path = self.directories['people'] / f"person_{person_uuid}.yaml"
                            
                            name = f"{selected.get('firstname', '')} {selected.get('lastname', '')}".strip()
                            
                            with open(yaml_path, 'w') as f:
                                yaml.safe_dump({
                                    'uuid': person_uuid,
                                    'contact': selected,
                                    'face_folders': [folder.name],
                                    'created_at': datetime.datetime.now().isoformat()
                                }, f, default_flow_style=False)
                            
                            # Update all face metadata in this folder
                            for face_file in folder.glob('*.jpg'):
                                if face_file.name.startswith('img_'):
                                    img_id = face_file.stem.split('_')[1]
                                    face_index = face_file.stem.split('_face')[-1]
                                    self.update_image_metadata(img_id, person_uuid, face_index)
                            
                            print(f"\nSuccessfully matched {folder.name} to {name}")
                            input("Press Enter to continue...")
                            break
                
                except Exception as e:
                    print(f"Error processing {folder.name}: {e}")
                finally:
                    # Close image window if possible
                    try:
                        img.close()
                    except:
                        pass
        
        print("\nFace matching complete!")

    def create_sample_contacts(self):
        """Create a sample VCF file with test contacts"""
        vcf_path = Path('contacts.vcf')  # Changed to root directory
        
        sample_contacts = [
            {
                'firstname': 'John',
                'lastname': 'Smith',
                'phone': '+1-555-123-4567',
                'email': 'john.smith@email.com'
            },
            {
                'firstname': 'Emma',
                'lastname': 'Johnson',
                'phone': '+1-555-234-5678',
                'email': 'emma.j@email.com'
            },
            {
                'firstname': 'Michael',
                'lastname': 'Brown',
                'phone': '+1-555-345-6789',
                'email': 'mbrown@email.com'
            },
            {
                'firstname': 'Sarah',
                'lastname': 'Davis',
                'phone': '+1-555-456-7890',
                'email': 'sarah.davis@email.com'
            },
            {
                'firstname': 'David',
                'lastname': 'Wilson',
                'phone': '+1-555-567-8901',
                'email': 'd.wilson@email.com'
            }
        ]
        
        try:
            with open(vcf_path, 'w') as f:
                for contact in sample_contacts:
                    f.write('BEGIN:VCARD\n')
                    f.write('VERSION:3.0\n')
                    f.write(f"N:{contact['lastname']};{contact['firstname']};;;\n")
                    f.write(f"FN:{contact['firstname']} {contact['lastname']}\n")
                    f.write(f"TEL;TYPE=CELL:{contact['phone']}\n")
                    f.write(f"EMAIL:{contact['email']}\n")
                    f.write('END:VCARD\n\n')
            
            print(f"Created sample contacts file with {len(sample_contacts)} contacts")
            
            # Also create corresponding YAML files
            for contact in sample_contacts:
                person_uuid = str(uuid.uuid4())
                yaml_path = self.directories['people'] / f"person_{person_uuid}.yaml"
                
                with open(yaml_path, 'w') as f:
                    yaml.safe_dump({
                        'uuid': person_uuid,
                        'contact': contact,
                        'face_folders': [],
                        'created_at': datetime.datetime.now().isoformat()
                    }, f, default_flow_style=False)
        
        except Exception as e:
            print(f"Error creating sample contacts: {e}")

    def create_person_yaml(self, contact_info, uuid=None):
        """Create a YAML file for a person with contact information if it doesn't exist
        Args:
            contact_info (dict): Dictionary containing contact information
            uuid (str, optional): UUID for the person. If None, generates new UUID
        """
        try:
            # Check if contact already exists by matching name and either email or phone
            existing_yaml = None
            for yaml_file in self.directories['people'].glob('person_*.yaml'):
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'contact' in data:
                        existing_contact = data['contact']
                        if (existing_contact.get('firstname') == contact_info.get('firstname') and 
                            existing_contact.get('lastname') == contact_info.get('lastname')):
                            # Additional check for email or phone match
                            if (existing_contact.get('email') == contact_info.get('email') or 
                                existing_contact.get('phone') == contact_info.get('phone')):
                                existing_yaml = yaml_file
                                break
            
            if existing_yaml:
                print(f"Contact already exists: {contact_info.get('firstname', '')} {contact_info.get('lastname', '')}")
                return None
            
            # Create new contact if doesn't exist
            if uuid is None:
                import uuid as uuid_lib
                uuid = str(uuid_lib.uuid4())
            
            person_data = {
                'contact': {
                    'firstname': contact_info.get('firstname', ''),
                    'lastname': contact_info.get('lastname', ''),
                    'email': contact_info.get('email', ''),
                    'phone': contact_info.get('phone', '')
                },
                'created_at': datetime.datetime.now().isoformat(),
                'face_folders': [],
                'uuid': uuid
            }
            
            yaml_path = self.directories['people'] / f"person_{uuid}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.safe_dump(person_data, f, default_flow_style=False)
            
            print(f"Created new person YAML file for {contact_info.get('firstname', '')} {contact_info.get('lastname', '')}")
            return uuid
            
        except Exception as e:
            print(f"Error creating person YAML: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_vcf_contacts(self, vcf_path):
        """Process VCF file and create YAML files for each contact
        Args:
            vcf_path (str): Path to VCF file
        """
        try:
            import vobject
            
            with open(vcf_path, 'r') as f:
                vcf_content = f.read()
                
            for vcard in vobject.readComponents(vcf_content):
                contact_info = {
                    'firstname': vcard.n.value.given if hasattr(vcard, 'n') else '',
                    'lastname': vcard.n.value.family if hasattr(vcard, 'n') else '',
                    'email': vcard.email.value if hasattr(vcard, 'email') else '',
                    'phone': vcard.tel.value if hasattr(vcard, 'tel') else ''
                }
                
                # Create YAML file for contact
                self.create_person_yaml(contact_info)
                
            print(f"Processed all contacts from {vcf_path}")
            
        except Exception as e:
            print(f"Error processing VCF file: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    processor = ImageProcessor()
    processor.process_all_images()
    processor.sort_similar_faces()
    processor.match_faces_to_contacts()
    processor.create_sample_contacts()




