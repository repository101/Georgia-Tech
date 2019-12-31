# Epsilon Photography

**Important Note:** This assignment is subject to the "Above & Beyond" rule. In summary: meeting all stated requirements will earn 90%; the last 10% is reserved for individual effort to research, implement, and report some additional high-quality work on this topic beyond the minimum requirements. Your A&B work must be accompanied by discussion of the computational photographic concepts involved, and will be graded based on the level of effort, and quality of your results, and documentation in the report. (Please review the full explanation of this rule in the syllabus and on Piazza.)

## Synopsis

Epsilon photography is a form of computational photography wherein multiple images are captured by varying a camera parameter such as aperture, exposure, focus, film speed, viewpoint, etc., by a small amount ε (hence the name) for the purpose of enhanced post-capture flexibility. (Wikipedia definition) The most critical aspect of epsilon photography is that *only one parameter changes* throughout the image sequence. 

For example, you may capture multiple images of a *stationary scene*:

  - At a specified interval between frames (e.g., a time-lapse)
  - Under different lighting conditions (e.g., flash/no-flash)
  - From different viewpoints
  - At different exposures
  - With different focal planes
  - At different apertures
  - With a *single* moving subject (e.g., stop motion with a single subject)
  - Many more! Be creative, but please remember the goals and restrictions of the assignment.

What NOT to do? Here are some examples from former students:

  - Do NOT submit pictures of your dog (pet) wearing different clothes and jumping around (maybe an example of epsilon fashionography, but not what we want!).
  - Do NOT submit one picture from each of your last 4 vacations. Vacation Photography, sure, but not Epsilon Photography.
  - Do NOT submit blurry stills extracted from a video taken with your smartphone.
  - Do NOT submit separate pictures of each member of your family in the same pose.
  - Do NOT submit pictures with light art where there are multiple changes (scene, light color, etc.)
  - Do NOT submit a series of pictures where each one has a _different_ small change (1st is exposure, 2nd is ISO, .... )

We are looking for evidence of intentional planning in your images. We want to see you plan what the images will be and how you can generate a new or novel view or image. Think about what your camera can do currently and how you can use computation to merge the resulting images into a novel result.

Some examples of Epsilon Photography from the past that went "above and beyond": 

   - Pictures of shaving one’s head, merged the aligned head pictures with different hairs to showcase an amazing final picture;
   - Smooth time lapse of a flower opening, a snail crawling (small change happening between images)


## Instructions

### 1. Create a sequence of 4-8 images using epsilon photography (we will refer to these as your "N pictures").

Your N pictures should have almost everything in common, and only **ONE thing varying** in very small (epsilon) amounts. (This may be challenging.) You should vary **ONLY one parameter** across all the images, not all parameters. And again, that parameter should vary in the smallest amount from one picture to another. 

Why do we repeat that it should be **only ONE parameter changing** so many times? Because misunderstanding this basic goal is the most common deduction on this assignment. It is critical that only one element changes. Any simultaneous changes in the scene, camera parameters, or camera orientation is not epsilon photography.

Most people will find they need a way to hold their camera still for this project because holding the camera by hand rarely works well for keeping a static scene. Using a support (improvised or tripod) generally produces better results. Small and/or cheap tripods exist, but are not required. (Although you may find them useful on later assignments, too.)

You may use any source code, open source tool, or commercial software (including Gimp, Photoshop, or other package software) to help remove minor camera shake by stacking and cropping, but you must note that you did this in your report and mention which software you used.


### 2. Create a final artifact using original Python 3 code

Combine your image sequence to produce a novel photographic artifact using **original code** written in Python 3. What can you do with these N images to generate a new image that shows a novel view or representation? Be creative. A couple of examples might include, but are not limited to, blending images together to show all changes at once, or stitching variations together into a unique pattern that artfully demonstrates your epsilon effect. Refer to Notebook 1 and opencv.org for examples on how to do this in Python and OpenCV. 
<br><b>Animated GIFs are NOT acceptable as a final artifact</b>. 

Add your N images (name `image1`, `image2`, etc) and final artifact (name `final_artifact`) to a file named `resources.zip` in the project folder. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

In `resources.zip`, also include a file named `README.txt` that contains basic instructions on running your code, including any prior setup needed with your N images, and what commands to use to run your code). This doesn’t need to be extremely detailed as long as we can successfully run your code based on your instructions.

You should NOT be using other people's source code, open source tools, or commercial software to be creating your final artifact. These are only okay for doing minor edits to your N images, like cropping. 


### 3. Complete the Report

Make a copy of the [report template](https://drive.google.com/file/d/1PZooY07fl-RrXKYeBw1NqdFiNPmn6Mi5/view?usp=sharing) and answer all of the questions in the template. Save your report as `report.pdf` in the project directory. 

### 4. Submit the Code

**Note:** Make sure that you have completed all the steps in [A0 - Course Setup](../A0-Course_Setup/README.md) for installing the VM & other course tools first.

Save your Python (.py) file as `epsilon.py`. Follow the [Project Submission Instructions](../A0-Course_Setup/README.md#3-submitting-projects) to upload your code to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI.


### 4. Submit the Report and Resources

Combine your `report.pdf` & `resources.zip` into a single zip archive and submit the file via Canvas. You may choose any name you want for the combined zip archive, e.g., `assignment1.zip`. Canvas will automatically rename the file if you resubmit, and it will have a different name when the TAs download it for grading. (In other words, you only need to follow the required naming convention for `report.pdf` and `resources.zip` inside your submission archive; don't worry about the name for the combined archive.) 

Example of Canvas submission hierarchy:
   - assignment1.zip
      - report.pdf
      - resources.zip
         - N input images
         - Final result 
         - README.txt

**Notes:** 

  - **DO NOT USE 7zip.** We've had problems in the past with 7z archives, so please don't use them unless you don't mind getting a zero on the assignment.

  - The total size of your project (report + resources) must be less than 30MB for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf).


## Criteria for Evaluation

Your submission will be graded based on:

  - Creativity, choice of domain, result quality, and a workflow that demonstrates Computational Photography.
  - Explaining your thought process as it relates to controlling your epsilon parameter change
  - Understanding of Epsilon Photography

If you turn in something with significant variations in each image or more than one change between frames, then you have NOT followed the instructions and the goal of this assignment and points will be deducted. 
