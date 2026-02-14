Note: Annotated output images would be generated when running the detection script on actual test images.

The detection_script.py will process input images and save:
1. Annotated images with bounding boxes (green for gloved, red for bare hands)
2. Confidence scores displayed on each detection
3. Images saved with same filenames as input

Example output naming:
- Input: factory_worker_001.jpg
- Output: factory_worker_001.jpg (with annotations)

To generate outputs, run:
python detection_script.py --input ./test_images --output ./output
