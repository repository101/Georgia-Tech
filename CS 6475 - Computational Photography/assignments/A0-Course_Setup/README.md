# CS6475 Computational Photography Course Setup

## Instructions

This is a preparatory assignment designed to help you setup your environment before starting A1. For this assignment, you will perform the following:
#### 1. Clone the course repository
#### 2. Setup the virtual environment
#### 3. Install the submission script (you will need this to submit your code for A1)
#### 4. Confirm you have access to the course Canvas and Piazza
#### 5. Become familiar with your camera's EXIF data




## 1. Cloning the Course Repository

Throughout the course assignments will be released via the course repository. To clone the course repository to your local machine, make sure you have [Git](https://git-scm.com/downloads) installed and the executable is on your system PATH. You can verify that you have Git installed and on your system PATH by executing the command `git --version` from a terminal. Clone the repository by issuing the command `git clone https://github.gatech.edu/omscs6475/assignments.git`.
    
Note: If you want to use the Git repository locally to track your changes, you should checkout a new branch immediately after cloning: `git checkout --b <new_branch_name>`

#### Receiving Updates

Updates can be retrieved by running `git pull` in the root directory of the repository from either your host or VM terminal. You should regularly pull and merge from the remote to ensure you have the most up-to-date commits.


## 2. Virtual Environment Setup

This course uses several third-party libraries (NumPy, SciPy, OpenCV, etc.) for projects and assignments. In order to standardize the execution of your code between your local environment and the remote autograding environment, we provide the specification for an Anaconda virtual environment that includes the correct version of all required software and libraries.

**NOTE: ALTHOUGH THE CONDA ENVIRONMENT HAS BEEN TESTED FOR CROSS-PLATFORM COMPATIBILITY WITH THE AUTOGRADER ENVIRONMENT, THE CONDA ENVIRONMENT IS NOT THE EXACT ENVIRONMENT YOUR CODE IS RUN IN BY THE AUTOGRADER. YOU ARE RESPONSIBLE TO ENSURE YOUR CODE WORKS ON THE AUTOGRADER SYSTEM–IT IS NOT ENOUGH THAT IT WORKS ON YOUR SYSTEM IN THE CONDA ENVIRONMENT.**

- Download and install the latest Python 3.5 supported version of [Anaconda](https://www.anaconda.com/download) for your OS (Windows/Linux/OS X). You may need to dig a little bit to find the [correct version](https://anaconda.org/anaconda/python/files) of Anaconda for the version of Python 3.5 you're using, but **it's important that you have Python 3.5 and the accompanying Anaconda build**.

- From the local course repository directory you created when you cloned the remote, follow the Anaconda [instructions](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to create the CS6475 virtual environment.

```
~/assignments$ conda env create -f cs6475.yml
```

You can then activate the virtual environment from a terminal by executing the following command:

```
~$ source activate CS6475
(CS6475) ~$
```

**NOTE:** Windows users should just use `activate CS6475` (you do not need the keyword `source`).

Once the virtual environment has been activated, you can execute code from the same terminal. It's worth mentioning that popular Python IDEs, such as PyCharm or Atom, facilitate using an Anaconda environment as the interpreter for your project or repository. This assignment will not go over those steps, as they will vary from tool to tool.

### Validate Environment Setup

You can test that your environment is setup correctly by opening a terminal, activating the virtual environment, and running the `test.py` file in the root folder of this repository. The test file performs basic checks to confirm that the required versions of all packages are correctly installed.

```
~$ source activate CS6475
(CS6475) ~$python3 test.py
.....
----------------------------------------------------------------------
Ran 5 tests in 0.272s

OK
```

**NOTE:** While we encourage students use the provided virtual environment, some prefer to manage their own environment. We will not provide support for students who choose to do so, but in any event, we are providing the list of libraries and their respective version numbers that you will use for the assignments. These libraries are packaged with the Anaconda environment, so only students who choose to use their own environment will need to adhere to this list. You will need the following libraries **and any dependencies**:
- Python 3.5
- MatPlotLib 3.0.0
- Nelson 0.4.2
- OpenCV 4.0.0.21
- NumPy 1.15.2
- SciPy 1.1.0

## 3. Submitting Projects

This repository includes a script to submit your code & projects to the autograder service (i.e., "[Bonnie](https://bonnie.udacity.com)").  You only need to install the script once (although you may need to reinstall if the repo changes). In this section, you will install the submission script, the `cs6475` CLI defined in `submit.py`:

 
  - Open a terminal and activate the course virtual environment, then navigate to the directory containing your copy of this repository.

```
~$ source activate CS6475
(CS6475) ~$
(CS6475) ~$ cd /assignments
(CS6475) ~/assignments$ 
```

**NOTE:** Windows users should just use `activate CS6475` (you do not need the keyword `source`).

  - Install the submission script `omscs` as an executable CLI script:

```
(CS6475) ~/assignments$ python3 -m pip install -e .
```

**NOTE:** Windows users should use `python.exe` instead of `python3`.

**NOTE:** Linux/Unix/OS X users may need to use "sudo" to install the script.


Once the CLI is installed, you can use it to upload your work to [Bonnie](https://bonnie.udacity.com). 

  - Obtain an [authentication token](https://bonnie.udacity.com/auth_tokens/new) from Bonnie. You should only need to perform this step once for the term; you can continue using the same jwt file for all submissions. However, you should **not** add your jwt file to your local git repo, nor upload it to the cloud. It is a secret key that authenticates your account to the Bonnie service.

  - Open a terminal, activate the virtual environment, and change your current working directory to the folder for the project you wish to submit.

```
~$ source activate CS6475
(CS6475) ~$
(CS6475) ~$ cd /assignments/A4-Blending
(CS6475) ~/assignments/A4-Blending$ 
```

  - To submit your code for grading, you will need to provide the path to the jwt file you downloaded above as `<jwt_path>`. The submission script can be executed with the following command:

```
(CS6475) ~/assignments/A4-Blending$ omscs -jwt <jwt_path> submit code

```

## 4. Canvas and Piazza

**Canvas** - If you haven't already, please check that you have access to CS-6475-001 on Canvas. You will use Canvas to submit your resources and report for assignments/projects. All assignment/project report submissions will only be accepted via Canvas, so it is very important to make sure that you have proper access. Message the Instructors on Piazza if you do not see the course. If you just enrolled in the course, it might take some time for you to see the course in Canvas, so give it a day or so.

**Piazza** - Make sure that you have access to the OMSCS version of the course Piazza. Use Piazza if you have any questions about the assignments/projects. Do NOT email TAs or the Professor directly unless it's for an urgent matter. Please make your question(s) public so all students can benefit from the discussions. Remember to use the appropriate tag for the assignment/project so it's easier to filter out questions. If you need to share your code or feel like you will give away an answer to an assignment question, please make your post PRIVATE to the Instructors only. Do NOT share your code or solutions to questions in the public forum. Our team will be monitoring Piazza and will remove posts that should not be shared publicly.

As mentioned in the course welcome, we will finalize the class roster at the end of the add/drop deadline at the end of the first week. If you logged into Piazza using a different email address than the GaTech one listed on our Course Roster, you will probably be dropped from Piazza, and you will have to rejoin. To avoid this:

Include your GaTech email and link multiple email addresses to your GaTech account under the Account/Email Settings, located by clicking on the gear on the top right of Piazza.  

## 5. Cameras and EXIF Data

You are allowed to use any kind of camera  -- anything from smartphones to high-end DSLR cameras -- for this class. Some assignments require granular control of aperture, shutter speed, ISO, and other settings that can be difficult to adjust for smartphones; other assignments require very stable positioning of the camera between successive shots that can be challenging without a tripod. However, there is no required hardware for this course -- many students successfully complete all of the assignments using only a smartphone.

To help you get used to your camera and its settings, you will be asked to provide some technical information about your photographs.  You should immediately find out how to get the EXIF data for the camera you plan to use in this course. You will need to be able to access:

  - Exposure time (ex: 1/2000 s, 1/30 s)
  - Aperture (ex: f 2.8, f16) 
  - ISO (ex: ISO 100, ISO 1600)

EXIF data is recorded in digital images, and can generally be found on your phone, in your digital camera, or on your computer. Image editing can affect the data, so record the settings before playing with your image.  Search online for information on finding EXIF data for your device if you are not sure how to find it or don’t know what it is. You can also discuss EXIF data and where to find it on Piazza.
```
