#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts reproduce all results found in the article.
"""

import subprocess
import argparse

def callcmd(cmd):
    pipes = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True)
    stdout, stderr = pipes.communicate()

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    
    # Add arguments to parse for training
    parser.add_argument("--run_commands",
                        type=int,
                        default=1,
                        help="If 1, runs the commands, else just print them"
                        )
                        
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()
     
    # 1.0 Training dataset creation on n GPUS
    print("##Creating training dataset")
    cmd = "python Case_article.py --training=0 "
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

    # 1.1 Perform the training for nmodel models
    print("##Training of the models")
    cmd= "python Case_article.py --training=1 --logdir=Case_article"
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

    # 1.2 Create the 1D test dataset
    print("##Creating the 1D test dataset")
    cmd= "python Case_article_test1D.py --testing=0 "
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

    # 1.3 Test on the 1D test dataset
    print("##Testing in 1D (Figure 3)")
    cmd = "python Case_article_test1D.py --testing=1"
    cmd+= " --logdir=Case_article*/4_* --niter=1000"
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

    # 1.4 Test in 2D
    print("##Testing in 2D (Figure 4)")
    cmd = "python Case_article_test2D.py --testing=1"
    cmd+= " --logdir=Case_article*/4_* --niter=1000"
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

    # 1.5 Test on real data
    print("##Testing on real data (Figure 5 and 6)")
    cmd = "cd realdata;"
    cmd+= "python Process_realdata.py"
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

    cmd = "python Case_article_test2D.py --plots=2"
    cmd+= " --logdir=Case_article*/4_* --niter=1000"
    print(cmd,"\n")
    if args.run_commands:
        callcmd(cmd)

