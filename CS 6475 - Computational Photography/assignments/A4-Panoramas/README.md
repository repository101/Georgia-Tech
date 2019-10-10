# Panoramas

**Important Note:** This assignment is subject to the "Above & Beyond" rule. In summary: meeting all stated requirements will earn 90%; the last 10% is reserved for individual effort to research, implement, and report some additional high-quality work on this topic beyond the minimum requirements. Your A&B work must be accompanied by discussion of the computational photographic concepts involved, and will be graded based on the level of effort, and quality of your results and documentation in the report. (Please review the full explanation of this rule in the syllabus or on Piazza.)


## Synopsis

In this assignment, you will be writing code to align & stitch together a series of images into a panorama. You will then use your code on your own pictures to make a panorama. Take at least 3 images (although feel free to take more) to use for a panorama. (You must take **your own** pictures for this assignment.) Refer to Udacity Lecture 5-03, and the course textbook (Szeliski chapter 9.1).


## Instructions

- Images in the `images/source/sample` directory are provided for testing -- *do not include these images in your submission* (although the output should appear in your report). 

- Downsampling your images to 1-2 MB each will save processing time during development. (Larger images take longer to process, and may cause problems for the VM and autograder which are resource-limited.) If your VM crashes due to a memory error then you may consider [increasing the RAM limit of the VM](https://github.gatech.edu/gist/cgearhart3/fef2306f0dace55abad5d73f6e849970) in your Vagrantfile. Shrinking your images is typically more effective than increasing the RAM limit.

- We have provided a basic script to execute your panorama pipeline by running `python main.py`. The script will look inside each subfolder under `images/source` and attempt to apply the panorama procedure to the contents, and save the output to a folder with the same name as the input in `images/output/`. (For example, `images/source/sample` will produce output in `images/output/sample`.)


### 1. Implement the functions in the `panorama.py` file.

  - `getImageCorners`: Return the x, y coordinates for the four corners of an input image
  - `findHomography`: Return the transformation between the keypoints of two images
  - `getBoundingCorners`: Find the extent of a canvas large enough to hold two images
  - `warpCanvas`: Warp an input image to align with the next adjacent image
  - `blendImagePair`: Fit two images onto a single canvas

**Note:** The `blendImagePair` function is not auto-scored. In order to receive _any_ credit for the function, you are required to replace the default insertion blend that we provide with your own blending function. Your resulting blend should show improvement over the default insertion blend function. We want you to be creative. Good blending is difficult and time-consuming to achieve. We do not expect you to implement a universally perfect and seamless blend to get basic credit for the function. You can also earn Above & Beyond credit for blending with technically challenging and creative solutions that exceed the minimum requirement of improving on insertion blending. 

The docstrings of each function contains detailed instructions. You are *strongly* encouraged to write your own unit tests based on the requirements. The `panorama_tests.py` file is provided to get you started. Your code will be evaluated on input and output type (e.g., uint8, float, etc.), array shape, and values. (Be careful regarding arithmetic overflow!) When you are ready to submit your code, you can send it to the autograder for scoring by running `omscs submit code` from the root directory of the blending project folder.


### 2. Generate your own panorama

Once your code has passed the autograder and you’ve run the test inputs, you are ready to assemble your own panorama(s).  Choose a scene for your panorama and capture a sequence of at least three (3) partially overlapping frames spanning the scene. Your pictures should be clear (not be blurry) for feature matching to work well, and you should make sure you have substantial overlap between your images for best results. You will need to explore different feature matching and keypoint settings to improve your results, and to record your modifications and parameter settings in your report (see the report template).


### 3. Above & Beyond (Optional)

Completing the basic requirements by implementing the required functions and generating a blended image with your pipeline will only earn at most 90% on this assignment. 10% of the assignment grade is based on (optional) "above & beyond" effort. (A&B credit is considered "optional" because your project is treated as "complete" as long as you meet the basic requirements.)

In order to earn A&B credit, you need to work independently (i.e., without instructor guidance) to extend this project in an interesting or creative way and document your work. It is up to _you_ to define the scope and establish the relevancy of your effort to the topic of blending images. You will earn credit on a sliding scale from 0-10% of the total project grade for things like creativity, technical difficulty, reporting on your work, and quality of your results.

Keep in mind:
- Earning the full 10% for A&B is typically _very_ rare; you should not expect to reach it unless your results are _very_ impressive.
- Attempting something very technically difficult does not ensure more credit; make sure you document your effort even if it doesn't pan out.
- Attempting something very easy in a very complicated way does not ensure more credit.


### 4. Complete the report

Make a copy of the [report template](https://docs.google.com/presentation/d/1aQJ4tO7KHccIoKZkGi4PapmMK1J5PQFI90Nq2CrslPw/edit?usp=sharing) and answer all of the questions. Save your report as `report.pdf` in the project directory.


### 5. Submit the Code

**Note:** Make sure that you have completed all the steps in the [instructions](../README.md#virtual-machine-setup) for installing the VM & other course tools first.

Follow the [Project Submission Instructions](../README.md#submitting-projects) to upload your code to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI:

```
$ omscs submit code
```


### 6. Submit the Report

Save your report as `report.pdf`. Create an archive named `resources.zip` containing your images and final artifact -- both files must be submitted. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

Combine your `report.pdf` & `resources.zip` into a single zip archive and submit the file via Canvas. You may choose any name you want for the combined zip archive, e.g., `assignment5.zip`. Canvas will automatically rename the file if you resubmit, and it will have a different name when the TAs download it for grading. (In other words, you only need to follow the required naming convention for `report.pdf` and `resources.zip` inside your submission archive; don't worry about the name for the combined archive.) YOUR REPORT SUBMISSION FOR THIS PROJECT DOES NOT NEED TO INCLUDE THE CODE, WHICH MUST BE SEPARATELY SUBMITTED TO BONNIE FOR SCORING.

**Note:** The total size of your project (report + resources) must be less than 15MB for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf).

**Note:** Your resources.zip must include your source images and your final panorama, as well as all images relevant to your A&B. If there isn't enough space, you may put your A&B input images and final panoramas in a folder on a secure site (Dropbox, Google Drive or similar) and include a working link in your report. Again, this only applies to your A&B images. Images that are part of the base assignment should be in resources.zip.


## Criteria for Evaluation

Your submission will be graded based on:

  - Correctness of required code
  - Creativity & overall quality of results
  - Completeness and quality of report


## Assignment FAQ

- Can we crop the resulting panorama in our report?

Yes, you may include a cropped panorama in the report, but you **MUST** also include the original uncropped version in your report. The functions we ask you to fill in expect uncropped image inputs and outputs.

- Can we add intermediate functions to our code for the blending portion of the assignment?

Yes, but it is your responsibility to make sure your code passes all tests on the autograder.

- Can I use a tool like Photoshop to improve the blending result? 

No, you may not use any other software; you must only use your own code to generate a final panorama. You are only allowed to use a simple image editing tool for cropping the final result (if you so choose).

- Can I use my blending function from the Pyramid assignment? 

Yes, you may use masks and pyramids to blend, but all code must be contained in your assignment file. Pyramid blending is resource intensive, and may cause problems on the VM unless you increase the resource limits from the Vagrantfile. (The autograder doesn't test the blending function, so it isn't an issue during grading.)

- Can I add extra python code files in my submission? 

No, all code you write for this assignment must be contained in the panorama.py file.
