# Dataset Preparation Guide for FLUX Training

## Overview

Proper dataset preparation is crucial for achieving 99.9% face accuracy with your RTX 5090 setup. This guide covers best practices for creating a high-quality training dataset.

## Dataset Structure

```
dataset/
├── image001.jpg
├── image001.txt
├── image002.jpg
├── image002.txt
├── image003.jpg
├── image003.txt
└── ...
```

Each image must have a corresponding `.txt` file with the same name containing the caption.

## Image Requirements

### Quantity
- **Minimum**: 10-15 images
- **Recommended**: 15-25 images
- **Maximum**: 50+ images (diminishing returns)

For face training:
- **Sweet spot**: 17-20 high-quality images

### Quality
- **Resolution**: Minimum 1024x1024 pixels
- **Format**: JPG or PNG
- **Aspect ratio**: Square (1:1) or close to it
- **Sharpness**: Clear, not blurry
- **Lighting**: Varied but consistent quality
- **File size**: 1-10 MB per image

### Diversity

For best results, include variety:

**Face Angles**:
- Front view (40%)
- 3/4 view (30%)
- Side profile (20%)
- Other angles (10%)

**Expressions**:
- Neutral (40%)
- Smiling (30%)
- Other expressions (30%)

**Lighting**:
- Natural light (50%)
- Studio/soft light (30%)
- Dramatic/side light (20%)

**Backgrounds**:
- Clean/simple (50%)
- Outdoor (25%)
- Indoor (25%)

**Distances**:
- Close-up (30%)
- Head and shoulders (50%)
- Partial body (20%)

## Caption Format

### Basic Format

For face training with trigger word:
```
triggerWord
```

Example (`image001.txt`):
```
t4r4woman
```

### Advanced Format (Optional)

For more control, you can add descriptive captions:

```
triggerWord, description
```

Examples:
```
t4r4woman, portrait photo, professional lighting
t4r4woman, smiling, outdoor natural light
t4r4woman, serious expression, studio background
t4r4woman, side profile, dramatic lighting
```

**Note**: For pure face learning, trigger-only captions often work better.

## Image Preparation Workflow

### 1. Collect Images

Sources (ensure you have rights to use):
- Personal photos
- Professional photoshoots
- Stock photos (with proper licensing)

**Avoid**:
- Heavily filtered images
- Low resolution images
- Watermarked images
- Multiple people in frame (unless that's the goal)

### 2. Curate and Select

Review all images and select the best:
- ✅ Clear face
- ✅ Good lighting
- ✅ Sharp focus
- ✅ Minimal distractions
- ❌ Blurry
- ❌ Heavily filtered
- ❌ Extreme angles
- ❌ Obstructed face

### 3. Crop and Resize

```bash
# Using ImageMagick (install: sudo apt install imagemagick)

# Crop to square and resize to 1024x1024
for img in *.jpg; do
    convert "$img" -resize 1024x1024^ -gravity center -extent 1024x1024 "processed_$img"
done
```

Or use a Python script:

```python
from PIL import Image
import os

def prepare_image(input_path, output_path, size=1024):
    """Resize and crop image to square"""
    img = Image.open(input_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Get dimensions
    width, height = img.size

    # Crop to square (center crop)
    if width > height:
        left = (width - height) // 2
        img = img.crop((left, 0, left + height, height))
    elif height > width:
        top = (height - width) // 2
        img = img.crop((0, top, width, top + width))

    # Resize to target size
    img = img.resize((size, size), Image.LANCZOS)

    # Save
    img.save(output_path, quality=95)

# Process all images
for filename in os.listdir('.'):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        prepare_image(filename, f'processed_{filename}')
```

### 4. Create Captions

Quick script to create caption files:

```bash
#!/bin/bash
# create_captions.sh

TRIGGER_WORD="t4r4woman"  # Change this to your trigger word

for img in *.jpg; do
    txt="${img%.*}.txt"
    if [ ! -f "$txt" ]; then
        echo "$TRIGGER_WORD" > "$txt"
        echo "Created: $txt"
    fi
done
```

Or Python version:

```python
import os

TRIGGER_WORD = "t4r4woman"  # Change this

for filename in os.listdir('.'):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        if not os.path.exists(txt_filename):
            with open(txt_filename, 'w') as f:
                f.write(TRIGGER_WORD)
            print(f"Created: {txt_filename}")
```

### 5. Verify Dataset

Checklist:
- [ ] All images are 1024x1024 (or consistent resolution)
- [ ] Every image has a corresponding .txt file
- [ ] All .txt files contain the trigger word
- [ ] No duplicate images
- [ ] 15-25 high-quality images total
- [ ] Good variety of angles and expressions

## Choosing a Trigger Word

### Guidelines

**Good trigger words**:
- Unique and unlikely to appear in random prompts
- Easy to remember
- Not a real word (to avoid conflicts)
- 6-12 characters

**Examples**:
- `t4r4woman` ✓
- `jhn5doe` ✓
- `s4r4hsmith` ✓
- `alex_unique` ✓

**Avoid**:
- Common words: `woman`, `portrait` ✗
- Single letters: `a`, `x` ✗
- Very long: `thisisaverylongtriggerword` ✗
- Special characters only: `@#$%` ✗

### Format Options

1. **Letters + Numbers**: `t4r4woman`
2. **Underscore**: `unique_person`
3. **CamelCase**: `JohnDoeAI`
4. **Abbreviated**: `jd2024`

## Advanced: Multi-Concept Training

To train multiple concepts in one LoRA:

```
dataset/
├── concept1/
│   ├── img001.jpg
│   ├── img001.txt  (contains: "trigger1")
│   └── ...
└── concept2/
    ├── img001.jpg
    ├── img001.txt  (contains: "trigger2")
    └── ...
```

Then in config, update:
```toml
train_data_dir = "./dataset"  # Points to parent directory
```

## Quality Assurance

Before training, run this verification:

```python
import os
from PIL import Image

def verify_dataset(dataset_dir):
    """Verify dataset is ready for training"""
    issues = []
    images = []

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(filename)

            # Check corresponding txt exists
            txt_file = os.path.splitext(filename)[0] + '.txt'
            if not os.path.exists(os.path.join(dataset_dir, txt_file)):
                issues.append(f"Missing caption: {txt_file}")

            # Check image dimensions
            img_path = os.path.join(dataset_dir, filename)
            img = Image.open(img_path)
            if img.size != (1024, 1024):
                issues.append(f"Wrong size: {filename} is {img.size}")

    print(f"Total images: {len(images)}")

    if len(images) < 10:
        issues.append(f"Too few images: {len(images)} (minimum 10)")
    elif len(images) > 50:
        print(f"⚠ Many images: {len(images)} (may take longer to train)")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Dataset ready for training!")
        return True

# Run verification
verify_dataset('./dataset')
```

## Example Dataset

For a person named "Tara":

1. **Trigger word**: `t4r4woman`

2. **17 images**:
   - 6 front-facing portraits
   - 5 three-quarter view
   - 3 side profiles
   - 2 other angles
   - 1 full body

3. **Captions**: All files contain just `t4r4woman`

4. **Result**: After 1500 steps, generates accurate faces of Tara

## Tips for Best Results

1. **Consistency is Key**
   - Use images from similar time period (same hairstyle, etc.)
   - Consistent image quality

2. **Avoid Over-Diversity**
   - Don't include drastically different looks
   - Keep makeup/style relatively consistent

3. **Quality > Quantity**
   - 15 excellent images > 50 mediocre ones

4. **Test and Iterate**
   - Start with 15-20 images
   - Train and evaluate
   - Add more if needed

## Common Mistakes

1. ❌ Using images from different decades
2. ❌ Including group photos
3. ❌ Low resolution source images
4. ❌ Heavily edited/filtered images
5. ❌ Inconsistent lighting quality
6. ❌ Too many accessories (glasses, hats, etc.)

## Ready to Train?

Once your dataset is prepared:

```bash
# Verify dataset
python verify_dataset.py  # Use script above

# Update config with your trigger word
nano config/rtx5090_native.toml

# Start training
./scripts/08_start_training.sh
```

---

**Next**: See [QUICKSTART.md](QUICKSTART.md) for training instructions
