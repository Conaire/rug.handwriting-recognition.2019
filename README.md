# #Keep_Scrolling Handwriting Recognition project 2019
Recognition of characters on handwritten Hebrew characters

## Virtual Environment (optional)
First install anaconda. Then create a virtual environment using the Anaconda prompt:
```bat
conda create -n hwr_test_env python=3.7
```
To activate the environment:
```bat
conda activate hwr_test_env
```

## Install dependencies
Install the project dependencies using pip:
```bat
pip install -r requirements.txt
```

## Running the code
To run the code run the following command:
```bat
python main_pipeline.py -f your/dataset/path
```
The text files will get generated in an output folder inside your dataset **your/dataset/path/output**

## Pipeline
* main_pipeline.py: the script that assembles all modules, from data importing to final .txt generation
  - preprocessing.py: the preprocessing module includes all preprocessing steps, by calling the corresponding scripts:
    - get_AOI.py: reading the images and determining useful area within them
    - noise_removal.py: noise removal performed using Non-local Means Denoising (function: _fast_nlmd_)
    - binarization.py: image binarization performed using a custom implementation of Chang's method<sup>1</sup> (function: _changs_method_)
    - TODO put in a function: focusing in on the text, removing surplus of white pixels
  - skewness_correction.py: skewness correction performed, where necessary, using Hough line transform (function: _hough_skew_correct_)
  - segmentation.py: line and character segmentation using watershed splitting (function: _segment_image_)
  - word_recognizer.py:
    - character/word recognition using neural network predictions (function: _recognize_word_)
    - verification/correction of predictions using Viterbi algorithm (function: _viterbi_for_word_)
  - document_converter.py: 
    - converting recognised characters to text (function: _convert_to_text_)
    - exporting recognised text into a .txt file (function: _convert_to_text_)
