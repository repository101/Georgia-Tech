# Panoramas

**Important Note:** This assignment is subject to the "Above & Beyond" rule. In summary: meeting all stated requirements will earn 90%; the last 10% is reserved for individual effort to research, implement, and report some additional high-quality work on this topic beyond the minimum requirements. Your A&B work must be accompanied by discussion of the computational photographic concepts involved, and will be graded based on the level of effort, and quality of your results and documentation in the report. (Please review the full explanation of this rule in the syllabus or on Piazza.)


## Synopsis

In this assignment, you will be writing code to align & stitch together a series of images into a panorama. You will then use your code on your own pictures to make a panorama. Take at least 3 images (although feel free to take more) to use for a panorama. (You must take **your own** pictures for this assignment.) Refer to Udacity Lecture 5-03, and the course textbook (Szeliski chapter 9.1).


## Instructions

- Images in the `images/source/sample` directory are provided for testing -- *do not include these images in your submission* (although the output should appear in your report). 

- Downsampling your images to 1-2 MB each will save processing time during development. (Larger images take longer to process, and may cause problems for the autograder which is resource-limited.)

- We have provided a basic script to execute your panorama pipeline by running `python main.py`. The script will look inside each subfolder under `images/source` and attempt to apply the panorama procedure to the contents, and save the output to a folder with the same name as the input in `images/output/`. (For example, `images/source/sample` will produce output in `images/output/sample`.)


### 1. Implement the functions in the `panorama.py` file.

  - `getImageCorners`: Return the x, y coordinates for the four corners of an input image
  - `findHomography`: Return the transformation between the keypoints of two images
  - `getBoundingCorners`: Find the extent of a canvas large enough to hold two images
  - `warpCanvas`: Warp an input image to align with the next adjacent image
  - `blendImagePair`: Fit two images onto a single canvas

**Important Notes:** The `blendImagePair` function is NOT auto-scored. You will see the autograder return a 0/10 for blendImagePair(), which is EXPECTED and if you implemented the other required functions correctly, you will receive a score of 30/40 from the autograder. We will be manually grading blendImagePair(), so there is no need to make sure your blendImagePair() implementation passes the autograder. Submit to the autograder and receive your 30/40 first before moving on to blendImagePair(). Your blendImagePair() must be implemented using Python and you may include additional imports if necessary. 

In order to receive _any_ credit for the blendImagePair() function, you are required to replace the default insertion blend that we provide with your own blending function. Your resulting blend should show improvement over the default insertion blend function. We want you to be creative. Good blending is difficult and time-consuming to achieve. We do not expect you to implement a universally perfect and seamless blend to get basic credit for the function. You can also earn Above & Beyond credit for blending with technically challenging and creative solutions that exceed the minimum requirement of improving on insertion blending. 

The docstrings of each function contains detailed instructions. You are *strongly* encouraged to write your own unit tests based on the requirements. The `panorama_tests.py` file is provided to get you started. Your code will be evaluated on input and output type (e.g., uint8, float, etc.), array shape, and values. (Be careful regarding arithmetic overflow!)


### 2. Generate your own panorama

Once your code has passed the autograder and you’ve run the test inputs, you are ready to assemble your own panorama(s).  Choose a scene for your panorama and capture a sequence of at least three (3) partially overlapping frames spanning the scene. Your pictures should be clear (not be blurry) for feature matching to work well, and you should make sure you have substantial overlap between your images for best results. You will need to explore different feature matching and keypoint settings to improve your results, and to record your modifications and parameter settings in your report (see the report template).


### 3. Above & Beyond (Optional)

Completing the basic requirements by implementing the required functions and generating a blended image with your pipeline will only earn at most 90% on this assignment. 10% of the assignment grade is based on (optional) "above & beyond" effort. (A&B credit is considered "optional" because your project is treated as "complete" as long as you meet the basic requirements.)

In order to earn A&B credit, you need to work independently (i.e., without instructor guidance) to extend this project in an interesting or creative way and document your work. It is up to _you_ to define the scope and establish the relevancy of your effort to the topic of blending images. You will earn credit on a sliding scale from 0-10% of the total project grade for things like creativity, technical difficulty, reporting on your work, and quality of your results.

**If you decide that your implementation for blendImagePair() is part of your A&B effort, make sure to say so in your blendImagePair() section of your report. Also, implementing a fantastic blend is not the only way to earn A&B credit. You may choose other things to explore that goes beyond the base requirements of this assignment.**

Keep in mind:
- Earning the full 10% for A&B is typically _very_ rare; you should not expect to reach it unless your results are _very_ impressive.
- Attempting something very technically difficult does not ensure more credit; make sure you document your effort even if it doesn't pan out.
- Attempting something very easy in a very complicated way does not ensure more credit.


### 4. Complete the report

Make a copy of the [report template](https://drive.google.com/file/d/1oUKWrXm5fRXcYp9fSirPC8Imb_EPjs0m/view?usp=sharing) and answer all of the questions. Save your report as `report.pdf` in the project directory.


### 5. Submit the Code

**Note:** Make sure that you have completed all the steps in [A0 - Course Setup](../A0-Course_Setup/README.md) for installing the conda environment & other course tools first.

Follow the [Project Submission Instructions](../A0-Course_Setup/README.md#3-submitting-projects) to upload your code `panorama.py` to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI.


### 6. Submit the Report

Save your report as `report.pdf`. Create an archive named `resources.zip` containing your images and final artifact -- both files must be submitted. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

**Note:** Your `resources.zip` **MUST** include your original input images (name `input_1`, `input_2`, etc.), your original panorama result (name `result`), and `panorama.py` that includes your original implementation of blendImagePair().  Submit your `report.pdf` & `resources.zip` **SEPARATELY** on Canvas. **DO NOT** zip `report.pdf` and `resources.zip` together. It is possible that Canvas will automatically rename the files if you resubmit, and it will have a different name when the TAs download it for grading. Do not worry about this if it happens. Please keep in mind that you are required to submit `panorama.py` to Bonnie for scoring (the max score the autograder will return is a 30/40). Again, `panorama.py` MUST be included in `resources.zip`in order for us to grade blendImagePair(), otherwise you will receive a 0 for this function.

**How to submit Above & Beyond work:** You are allowed to include your A&B work (input images, final results, code files) in `resources.zip` as well. You may name your A&B files how you want, as long as it's clear they are your A&B work (ex: `input1_AB.png`, `result_AB.png`). If there is not enough space (i.e. if you're about to go over the required size limit (see below)), you may put your A&B images in a folder on a secure site (ex: Dropbox) and include a working link in your report. Again, this only applies to your A&B images. Your images from the base assignment MUST be in `resources.zip`. Keep in mind that for A&B, we cannot grade what we cannot see, so if you state in your report that you implemented something yourself, but do not show us the code, you will not be awarded additional points. Make sure you submit your A&B code files in `resources.zip`. 

**Submission Size:** The total size of your project (report.pdf + resources.zip) **MUST** be less than **15MB** for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf).

Example of Canvas submission hierarchy:
   - `report.pdf`
   - `resources.zip`
      - Input image 1 (ex: `input_1.jpg`)
      - Input image 2 (ex: `input_2.jpg`)
      - Input image 3 (ex: `input_3.jpg`)
      - Final panorama (ex: `result.jpg`)
      - `panorama.py`
      
**Notes:**

  - When sharing images, make sure there is no data contained in the EXIF data that you do not want shared (i.e. GPS). If there is, make sure you strip it out before submitting your work or sharing your photos with others. Normally, we only require that your submitted images include aperture, shutter speed, and ISO (not a requirement for A4).

  - **DO NOT USE 7zip.** We've had problems in the past with 7z archives, so please don't use them unless you don't mind getting a zero on the assignment.


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

Yes, you may use masks and pyramids to blend, but all code must be contained in your assignment file. Pyramid blending is resource intensive, and may cause problems. (The autograder doesn't test the blending function, so it isn't an issue during grading.)

- Can I add extra python code files in my submission? 

No, all code you write for this assignment must be contained in the panorama.py file.
