# Research Project

## Synopsis

The goal of this project is to replicate the results of a previously published computational photography paper. Replication provides a deeper understanding of research results, and allows you to compare your results against an expected baseline. While previous results are helpful as a guide towards implementation, successful replication is often challenging. Instructions are not always clear and implementation details like parameter values may be missing or ambiguous, which makes it sometimes difficult or even impossible to achieve exactly the same results.

One of the major goals of this project is for you to work through the research paper in the face of uncertainty. To that end, the TAs will only answer logistical questions about this project; they will not answer questions about the research paper or the algorithm itself. If the description is imperfect then you will need to make assumptions and experiment to find an implementation that produces results that you find acceptable. You are strongly encouraged to discuss the paper with your classmates on Piazza to work through it, but do **not** share code.

**Please remember that you may not use any part of this algorithm from another source in print or online—doing so is PLAGIARISM.**

**START EARLY.**

This is the first project with significant coding required. Historically, students have report spending _much_ more time on this project than any of the earlier projects in the term, and students often report that their initial implementations of this algorithm can take _hours_ to run (which makes experimentation and debugging very slow). Keep in mind that you will not be graded on the performance of your implementation, and your code is not expected to operate in real time.

Performance Tips:
- You are not required to use Python, so using a compiled language may result in faster runtimes (although it may take more time to write if you aren't already familiar with image processing in a compiled language); also, Python with good profiling/optimization can be reasonably fast
- Consider writing intermediate and final result matrices to disk for manual inspection or iterative development to save time
- You may be able to improve runtime by profiling your code and replacing bottleneck functions with cython, or using a JIT like [numba](http://numba.pydata.org/)
- Vectorize your Python code wherever possible; you may use built-in functions in third-party libraries (including numpy/scipy/openCV) for efficient array operations as long as they do not directly implement the seam carving algorithm
- You can iterate and test faster by making smaller copies of the input images to test your code, but your final results must be generated using the full size images provided in the git repository


## Instructions

Your assignment is to replicate the published results of [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html) by Shai Avidan and Ariel Shamir. (There is a PDF copy of the paper in the course resources on Canvas.)

You will deliver:

  - Source code containing **your own** implementation of the seam carving algorithm (you *may not* use any part of any existing implementation in your submission)

  - A 2-3 page written report (**3 page max**)

  - A short video presentation (**3 minutes max**)

**NOTE:** We are serious about the 3 pages and 3 minutes limit. A report of 3 pages and *one* line will cost you points. A video of length *3:01* will cost you points.


You must reproduce the results for:

  - Seam removal (Figure 5 -- you do **not** need to show scaling or cropping)

  - Seam insertion (Figure 8 -- parts c, d, and f only)

  - Implement optimal retargeting with dynamic programming (Figure 7 -- show the transport map with your seam removal path & the result of optimal retargeting; you are not required to show the other three versions using alternating row/col seam removal)


Your report must include:

1. The replicated images for the required parts of Fig 5, Fig 8, and Fig 7
2. A brief description of the algorithm (including an overview of your implementation and description of all important functions)
3. Compare your results to those in the original paper (including any significant differences you observe); you can find high-quality copies of some inputs and outputs from the research paper on the author's website [here](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html) (note: the high quality copies of the butterfly images are not available)
4. List and discuss at least three things from the original paper that made the results difficult to replicate (e.g., ambiguous instructions or descriptions, unclear assumptions, etc.) and explain how you overcame them.

Your video must include:

There are no specific requirements for the video. Some students use the video to demonstrate running their code, some students make a presentation summarizing interesting observations that don't fit in their reports, and some students focus on specific issues or topics that they found interesting–these (and other ideas) are all acceptable.

**NOTES:**

- **Details on the code:**
  - You may use any language of your choice as long as you provide complete instructions for the TAs to replicate your outputs.
  - You may use 3rd party libraries for image file I/O, linear algebra support, and primitive image operations (matrix arithmetic, finding min/max elements of matrices, gradient operators, etc.).
  - You **MAY NOT** use any existing implementation of the seam carving algorithm as a starting point; you must write **your own** implementation. *Including any part of the algorithm from any other source is plagiarism.*
  - Your submission must include all source code, support libraries, and a README file with *complete* instructions to execute your code. ("Complete" means that every required step must be explicit in your instructions; e.g., do NOT say "install `customImageLibrary`", instead give the **exact** command -- we should be able to copy/paste your instructions to get set up.)

- **Details for the report:**
  - The report must be in PDF format; no other format will be accepted.
  - There is no required template for the report; however, the format should be approximately equivalent to something like AAAI or NIPS format (see the [templates](/MT-Research_Project/templates) folder for reference).  *You are NOT required to use TeX; the templates are provided for reference.*
  - If you do not use one of the provided templates, font size should be at least 10pt and margins at least 1"; single-spaced, single column.
  - You do not need a title page, cover page, etc.
  - Include a link to your video in your report.

- **Details for the video:**
  - The video must be uploaded to an internet hosting platform.
  - Make sure that the link is not searchable, and that it is shared properly (i.e., only you and people with the link should be able to access it).
  - Do not speed up your presentations to compress the time.


## Submit the Project

Save your report in the project directory as `report.pdf`, and put your source code and supporting files (including your output images) in a file named `resources.zip` -- both files are required.  If your code has library dependencies or setup instructions that are required, then your resources file **must** include instructions in a plaintext file named `README.txt`. You will not submit your video file; it must be hosted on YouTube, google drive, dropbox, or similar. Add your report & resources files into a single zip archive (DO NOT USE 7zip!) and submit the zip file in Canvas for this project. The name of your zip file does not matter. Canvas automatically renames uploaded files in some cases, so you can use any name—for example `MTProject.zip` or similar.

**ATTENTION:** Make sure you include a link to your video clearly in your report. Also make sure you have permissions set to allow TAs to access your video or there will be a substantial deduction.

**Note:** The total size of your project (report + resources) must be less than 30MB for this project. If your submission is too large, you can reduce the scale of your images or report.


## Evaluation Criteria

Your work will be graded on:
  - Meeting all the specified requirements for the code, report, and presentation
  - Your understanding and communication of the algorithm and results from the original paper
  - The quality of your replication results
  - The quality of your report & presentation
