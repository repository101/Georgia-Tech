#!/usr/bin/env python3
import argparse
import json
import os
import textwrap

from zipfile import ZipFile

import nelson.gtomscs as nelson

import requests

try:
    requests.packages.urllib3.disable_warnings()  # try to disable urllib3 warnings
except AttributeError:
    pass


COURSE_POLICIES = """\
Course Policies:

    I have read the Computational Photography syllabus and understand
    all course policies, including:

        - Submissions are limited to three times every 2 hours
        - Only my LAST submission will be graded
        - Accidentally resubmitting an assignment during the late period
          is considered a late submission
        - The full late penalty deduction applies to the entire assignment
          score if any part of an assignment is submitted late
"""

HONOR_PLEDGE = """\
Honor Pledge:

    I have neither given nor received aid on this assignment.
"""


def require_pledge(policy_text):
    print(policy_text)
    ans = input("Please type 'yes' to agree and continue>")
    print
    if ans.lower() == "yes":
        return True
    return False


def submit(course_id, args, settings):
    required_files = settings.get("required_files", [])
    optional_files = settings.get("optional_files", [])
    user_files = args.filenames
    filenames = set([f for f in required_files + optional_files + user_files if os.path.isfile(f)])

    # check for redundant files in resources.zip
    if "resources.zip" in filenames:
        with ZipFile('resources.zip', 'r') as archive:
            zipnames = archive.namelist()

        redundant_files = [f for f in filenames if f in zipnames]
        if redundant_files:
            print(textwrap.dedent("""\
            ******************************************************************
                                   Submission Failed                          
            ******************************************************************

            Your resources.zip archive includes redundant files:
            {}

            Remove these files from resources.zip and try again.
            """).format('\n'.join(['    - {!s}'.format(f) for f in redundant_files])))
            exit()

    if not set(required_files) <= filenames:
        print(textwrap.dedent("""\
        ******************************************************************
                               Submission Failed
        ******************************************************************

        One or more required file(s) were missing:
        {}

        Make sure these files exist in the current directory and try again.
        """).format('\n'.join(['    -{!s}'.format(f) for f in set(required_files) - filenames])))
        exit()

    try:
        nelson.submit(course_id, settings["quiz_id"], filenames,
                      environment=args.environment,
                      max_zip_size=settings.get("size", 8) << 20,
                      jwt_path=args.jwt_path,
                      zipfile_root='')
    except ValueError as e:
        print(textwrap.dedent("""
        
        ******************************************************************
                               Submission Failed
        ******************************************************************

        {!s}

        """).format(e))
    except RuntimeError as e:
        print(textwrap.dedent("""
        
        ******************************************************************
                               Submission Failed
        ******************************************************************

        {!s}

        """).format(e))
        print("{:^70}\n".format("REMINDER: REPORTS MUST BE SUBMITTED VIA CANVAS"))
        exit()


def main(policies=[COURSE_POLICIES, HONOR_PLEDGE]):
    try:
        with open(".bonnie", 'r') as f:
            settings = json.load(f)
    except IOError:
        print("Error: No file named `.bonnie` found -- make sure you are in a valid project directory.")
        exit()

    aparser = argparse.ArgumentParser(
        description='Submit files to GT Autograder.')
    aparser.add_argument(
        '-jwt', '--jwt_path', default=None,
        help="Path to authentication token.")

    subparsers = aparser.add_subparsers(
        dest="action",
        help="Choose an action for the submit script to execute"
    )

    sparser = subparsers.add_parser("submit")
    sparser.add_argument(
        'quiz', choices=settings["quizzes"].keys(),
        help="Select the name of the quiz to submit")
    sparser.add_argument(
        '-f', '--filenames', nargs="+", default=[],
        help="The names of any additional files to submit.")
    sparser.add_argument(
        '-e', '--environment', default='production',
        choices=['development', 'production', 'staging'],
        help="Select the server to use (default: production)")
    args = aparser.parse_args()

    for policy_text in policies:
        if not require_pledge(policy_text):
            print("Error: You must accept all policies before submitting your assignment.\n")
            exit()

    submit(settings["course_id"], args, settings["quizzes"][args.quiz])


if __name__ == '__main__':
    main()
