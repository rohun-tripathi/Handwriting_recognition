### Handwritten OCR

A handwriting recognition project that `just works`.
It takes as input an English handwritten word and gives you the prediction of the word.
It processes data at the character level and can be used for character level predictions, if required.

The model here has two levels of outputs. The first level provides output combined at the character level. The second level runs the output through a large dictionary and chooses the words within one or edit distances that have a higher probability of occuring. Using the corpus, a user can bias the output to extract keywords from the domain they want to focus on.

The Handwriting model uses a pre-trained model available on the CRNN repository for text recognition in the wild - https://github.com/bgshih/crnn to initialize its weights.

The final trained model is performs well for both, handwriting recognition and text recognition in the wild.

#### Examples:
![alt text](https://github.com/rohun-tripathi/Handwriting_recognition/blob/master/crnn.pytorch/data/top.png?raw=true)
-> 'stop'

![alt text](https://github.com/rohun-tripathi/Handwriting_recognition/blob/master/crnn.pytorch/data/meeting.png?raw=true)
-> 'merhing' (output at character level)

-> 'meeting' (output after post processing using the corpus)


![alt text](https://github.com/rohun-tripathi/Handwriting_recognition/blob/master/crnn.pytorch/practice_demo/Screen%20Shot%202017-12-11%20at%202.24.58%20AM.png?raw=true)
-> 'sanding' (output at character level)

-> 'standing' (output after post processing using the corpus)

#### Ways to test:

(1) Run : `python crnn.pytorch/demo.py`

(2) Start the flask app and upload an image via the interface on localhost:5000, or using -
    
        python crnn.pytorch/flaskr.py
        curl --data "index=1" http://localhost:5000/image_txt
        curl --data "keyword=spinal, disease, wound" http://localhost:5000/search_txt
        curl -X POST --data "{image:base_64_embedding_of_word_image}" http://localhost:5000/single_image

#### Motivation
There are huge repositories of archived handwritten data that needs to be transcribed for archivists to be able to digitally curate them.
We are talking millions of documents per public library. 

#### Installation

Clone this repo. Install PyTorch. The code converts the GPU trained models to CPU models, and that same code can be used for other such conversions.

#### Data Sets

This repo relies on the IAM Offline English word dataset - http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database
Please download the word level images and extract them inside `./crnn.pytorch/data` and the directory name should be `words`.

#### Contribution
Possible directions is making it support more languages or making it a more robust English model.

#### Performance
It achieves 66.73% exact word match accuracy on the IAM test data set if using the character level layer output.
The test set was created by holding out 25% of the complete data set.

It achieves 73.90% exact word match accuracy on the same IAM test data set if using output after the post processing layer.

For training with variable length images, please sort the image according to the text length. This is not breaking, and model trains well without it. But it can help increase a few percentages

#### Usability
This project contains a bootstrap based UI that can be used to demonstrate the validity of the project and for quick start.

##### The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
