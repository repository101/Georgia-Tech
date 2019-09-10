# Camera Obscura

**Important Note:** This assignment is subject to the "Above & Beyond" rule. In summary: meeting all stated requirements will earn 90%; the last 10% is reserved for individual effort to research, implement, and report some additional high-quality work on this topic beyond the minimum requirements. Your A&B work must be accompanied by discussion of the computational photographic concepts involved, and will be graded based on the level of effort, and quality of your results and documentation in the report. (Please review the full explanation of this rule on Piazza.)


## Synopsis

In this assignment, you will construct a Camera Obscura (also known as a pin-hole camera), and document the construction and operation of the camera. The camera obscura is the forerunner of modern cameras, as shown in this [drawing of the Arabic scientist Al-Hasan in the 10th century](https://padumedu.files.wordpress.com/2013/12/alhazen.jpg) and in this [17th century manuscript](https://upload.wikimedia.org/wikipedia/commons/8/8a/Camera_obscura2.jpg). The best way to learn about this first primitive camera and understand how it serves as a basic model of all cameras is to experience it yourself. Hence, this assignment. There is no coding for this assignment, and so the submission will consist purely of submitting your PDF report and resources.

#### General Advice

Part of your task is to experiment to find the best setup.

  - The outside scene should be in bright daylight.
  - Choose a very dark room and relatively small pinhole (ranging from ⅛” to coin sized, depending on room size).
  - Control the light entering the room. (Heavy black plastic, blankets, aluminum foil, and cardboard are frequently used.)
  - If your walls have dark paint, or the image is very dim, improvise a white screen closer to the pinhole to reveal the image.
  - If your room or camera absolutely won't work for this task, you can try to borrow a camera or room from a friend or family member.
  - Experiment with different pinhole diameters (this is a requirement), especially if the image is really dim.
  - Keep your camera still during the entire exposure time (tripod or other mounting). Use your camera’s timer to help - a delay gives you time to set the camera and turn off room lights, and removes a source of motion.
  - You may do some editing/enhancement to brighten the image or flip it, but keep notes on the procedure you followed to take the picture and make sure you discuss it in your report.
  - Smartphones often have difficulties taking long exposure images. Students have found slow shutter apps that have worked well for them. As apps change constantly, we do not recommend any particular ones, but feel free to discuss with other students on Piazza.


#### Examples

  - You can see  a couple examples of scenes, pinholes, and final images previous students took on Canvas at: Files>A3 - Sample Images>CS6475-A3-SamplesFromPastStudents.pdf. This is NOT a model sample of the report.

  - For discussion of box obscura and some results see http://www.janetneuhauser.com/thoughts-on-lensless-photography/

Please note there may be differences between these examples and the requirements of this assignment. The examples are meant to highlight the types of images and ways that previous students have been able to capture in relation to this assignment, but please follow the directions in this assignment.


## Instructions

This video gives a good overview of the process you will follow: [Making Your Own Room With a View](https://www.youtube.com/watch?v=gvzpu0Q9RTU)

### 1. Build a Camera Obscura

Convert a room in your house (or a closet, or any small enclosed place) into a camera obscura by letting light pass into the room only through a small hole (i.e., the pinhole). Block out all light from windows, doorways, and any other sources, leaving only a small pinhole in one surface. An image of the outside world will be projected onto a wall in your room. The image will be dim but visible to the naked eye (once your eyes acclimate to the darkness) as long as the space is very dark.

You may get help from a friend or family member, but we expect that you will direct the efforts and document what they did to help. Document your efforts by taking pictures, even for setups that fail. These are lessons learned that you should include in your report.

*NOTE:* It is a good idea to let your neighbors know what you are doing. Covering windows with cardboard and foil has occasionally caused consternation. You may share the class website with them: https://www.omscs.gatech.edu/cs-6475-computational-photography


#### Alternative Box Camera Obscura

You must make a serious attempt to building a camera obscura in a room as described above before building a box camera. However, if you cannot use a room, then you may construct a box camera. You must explain in your report why you could not build a room and you will not be penalized.

A box camera obscura can be built using a large cardboard box or other light-proof container. Consider a moving box, paper supply box, or grocery store boxes. If you follow this approach, you will not get the true experience of being inside a camera obscura, but a box can still give very good results. The advantage of this method is portability, so you can make images of a variety of outdoor scenes.

If you choose this approach, investigate how to see the image inside and how to capture a picture of the interior. This is challenging, but there are solutions. In the past, students have achieved good results by taking a picture of the back/side screen of the box with their camera looking through a view-hole cut adjacent to the pinhole -- but the camera location will need light-proofing. Another option is to seal the camera inside the box on a timer or remote control.

If you choose this alternative approach, then make sure that you:

  - Document in your write-up the steps that you took to try the room camera obscura, and/or what prevented you from being successful in using a room for the assignment. This can include lack of suitable location, lighting problems, weather conditions, etc.

  - Make a light-proof box or container. (This takes care and effort!) You will need to capture the image projected onto the back of the box with the box closed to prevent light leakage.

  - The image through the pinhole should project onto a solid opaque surface at the back of the box. This means that the light can’t shine through the back of the box. You should place a white screen along the box walls, as images don’t show up well on brown cardboard.

  - Try various pinhole sizes, but start small for boxes - **See report template for instructions on the Pinhole Experiment part of this assignment**

  - Capture the building process of your box camera obscura. Take pictures as you go so you can use them in your report.


#### What **NOT** to do (there will be a substantial deduction for any of these):

  - Do NOT build a box camera where you take pictures from behind a translucent screen. This design blurs or “fogs” the image considerably, and generally has substantial light leakage.  An example of this excluded type of camera can be viewed at this youtube video https://www.youtube.com/watch?v=Y0wenfVfHuo.

  - Do NOT build a pinhole lens for your camera. The simplest way to get a Pinhole Camera for those with a removable lens camera is to just make a small hole in a lens-cover and put it in front of the camera. This IS NOT ACCEPTABLE. The assignment requires a room, or the alternative box/container, where you can make the box into a camera with a screen.

  - Do NOT simply stick the camera through the back of a box camera and photograph the pinhole.

  - Do NOT use a lens for the basic assignment. You may investigate using a lens and compare between pinholes and a lens after using the plain pinhole camera, but you must submit results from a camera obscura with just a pinhole lens.


### 2. Document your work

  - Take a picture of the scene outside the window (on the other side of the pinhole) that you are trying to capture (let’s call it the "Scene").
  - Take a picture of the image being projected onto your wall or screen (let's call it the "Image") by using a very long exposure (for example, from 8 to 30 seconds).
  - Take one or more images of the final setup, (let’s call this the "Setup"). This, at a minimum, should include an image showing the overall room and the covered window (with the pinhole).

**Keep in mind that you will need to take pictures for the pinhole experiment part of this assignment as well. Please read the report template first before taking down your setup.**

If you have code or images that you want to share outside of your report, add them to a file named `resources.zip` and place it in the project repository.


### 3. Above & Beyond (Optional)

Completing the basic requirements by building a camera obscura will only earn at most 90% on this assignment. 10% of the assignment grade is based on (optional) "above & beyond" effort. (A&B credit is considered "optional" because your project is treated as "complete" as long as you meet the basic requirements.)

In order to earn A&B credit, you need to work independently (i.e., without instructor guidance) to extend this project in an interesting or creative way and document your work. It is up to _you_ to define the scope and establish the relevancy of your effort to the topic of cameras and optics. You will earn credit on a sliding scale from 0-10% of the total project grade for things like creativity, technical difficulty, reporting on your work, and quality of your results.

Keep in mind:
- Earning the full 10% for A&B is typically _very_ rare; you should not expect to reach it unless your results are _very_ impressive.
- Attempting something very technically difficult does not ensure more credit; make sure you document your effort even if it doesn't pan out.
- Attempting something very easy in a very complicated way does not ensure more credit.


### 4. Complete the Report

Make a copy of the [report template](https://drive.google.com/file/d/1chGTnMAjuc6u4Zx00dRuElKO1K5asNwH/view?usp=sharing) and answer all of the questions. Save your report as `report.pdf` in the project directory. Make sure to document and report if you have any failed setups and include them in your discussion section of the report.


### 5. Submit the Report

Save your report as `report.pdf`. Create an archive named `resources.zip` containing your images (scene, setup, final image(s)) and any additional files/images you want to submit. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

Combine your `report.pdf` & `resources.zip` into a single zip archive and submit the file via Canvas. You may choose any name you want for the combined zip archive, e.g., `assignment2.zip`. Canvas will automatically rename the file if you resubmit, and it will have a different name when the TAs download it for grading. (In other words, you only need to follow the required naming convention for `report.pdf` and `resources.zip` inside your submission archive; don't worry about the name for the combined archive.)

Example of Canvas submission hierarchy:
   - assignment2.zip
      - report.pdf
      - resources.zip
         - Scene image
         - Setup images 
         - Final image(s)

**Important Notes:**

  - When sharing images, make sure there is no data contained in the EXIF data that you do not want shared (i.e. GPS). If there is, make sure you strip it out before submitting your work or sharing your photos with others. We only require that your images include aperture, shutter speed, and ISO.

  - **DO NOT USE 7zip.** We've had problems in the past with 7z archives, so please don't use them unless you don't mind getting a zero on the assignment.

  - The total size of your project (report + resources) must be less than 8MB for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf).
  
  - Your resources.zip must include your scene, setup, and final images, as well as all images relevant to your A&B. If there isn't enough space, you may put your A&B images in a folder on a secure site (ex: Dropbox) and include a working link in your report. Again, this only applies to your A&B images. Images that are part of the base assignment should be in resources.zip.


## Criteria for Evaluation

We are aware that this is **NOT** an easy assignment, but it is designed to give you insights about cameras "the old-fashioned way", and also about planning and capturing images. The pinhole camera model is fundamental to understanding the geometry associated with photography.

We are also aware that this assignment reminds some of you of your early schooling days, but, overall, every class since 2008 that has done this assignment has enjoyed it. If you find this assignment to be too “easy”, then put extra effort into your analysis of the the pinhole geometry model and A&B work. Reviewing the textbook and performing web searches will provide plenty of opportunity to further explore the mathematics associated with a camera obscura. Start early and plan ahead.

This assignment will be open for peer-feedback. Please do not share any images you don't want anyone to see. 

Your submission will be graded based on:

  - Creativity, result quality, and a workflow that demonstrates forethought & planning
  - Results that demonstrate a technical understanding of the pinhole camera model


## Appendix: Field of View

![Field of View](FOV.png)

The field of view (FOV) for a camera is an angle (measured in degrees). You can measure the angle with a protractor (you can find printable ones on the web if you don’t have one), or calculate the angle based on measurements in your camera room. Try to make your measurements reasonably accurate.

For the camera obscura, the included angle is measured using the pinhole as the vertex location from the first place that an image is visible on one side of your room or box, to the last place you can view the image on the other side. We are only requesting the horizontal field of view, however you may also supply the vertical if you wish. Note that the FOV is NOT limited to the area that you may be using for a screen; rather it may continue along your walls. Use a piece of paper to track the image, if necessary.

The FOV for the digital camera that you are capturing the image with is slightly different. We assume that you placed your camera near your pinhole, and used whatever settings worked to capture the image. You will need to look at your full camera image. If you know the camera position, this can be done with the lights on.  Measure the angle between the last object visible at the two side edges of your image using the camera lens position as the vertex.
