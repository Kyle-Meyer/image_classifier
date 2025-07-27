#!/usr/bin/env python3

#######################################################################
##  You will need:                                                    #
##    a) python3                                                      #
##    b) numpy                                                        #
##                                                                    #
## HOW: python3 en626BestMatch_py3.py raw_features.csv 4 5            #
##              (where 4 corresponds to the image in line 4 of        #
##               the raw_features.csv file)                           #
##                                                                    #
##  Input:                                                            #
##    a) Raw (not normalized) .CSV file with all the features         #
##                                                                    #
##  Output:                                                           #
##    b) Top matches                                                  #
#######################################################################

import sys
from subprocess import *
import numpy as np

NUM_MATCHES = 12

##########################################################
##  Read CSV file                                       ##
##    Input: input file name                            ##
##    Output: list with names (all_names)               ##
##            array with all the featurs (all_features) ##
##########################################################
def read_array(filename, separator=','):
    f = open(filename)
    line = f.readline()

    array = []
    name_array = []

    #Loop thru the file
    i=0
    while line:
       line = line.rstrip("\n")
       line = line.split(separator)

       #Create new row for features
       array.append([])
       for feature in line[1:]:
           array[i].append(float(feature))

       #store name
       name_array.append(line[0])

       line = f.readline()
       i+=1

    #Convert python list to Numpy array
    narray = np.array(array)

    return name_array,narray


##################################
##  Normalize all the features  ##
##################################
def normalize_data(all_features):
   #Get index of max and min values within each column
   row_index_max = all_features.argmax(axis=0)
   row_index_min = all_features.argmin(axis=0)

   rows,columns = all_features.shape

   normalized_features = np.zeros((rows,columns), float)

   #Create a 2D with ranges 
   #    min of feature1 == feature_range[0][0]
   #    max of feature1 == feature_range[0][1]
   feature_range = []
   for c in range(columns):
      feature_range.append((all_features[row_index_min[c]][c], all_features[row_index_max[c]][c]))

   return feature_range, normalize_array(all_features, feature_range)


##################################
##  Normalize all the features  ##
##################################
def normalize_array(all_features, feature_range):
   rows,columns = all_features.shape
   normalized_features = np.zeros((rows,columns), float)

   #Loop thru each feature and normalize it between 0 and 1
   for r in range(rows):
      for c in range(columns):
         if abs(feature_range[c][1] - feature_range[c][0]) > 0.0001:
            normalized_features[r][c] = (all_features[r][c] - feature_range[c][0]) / (feature_range[c][1] - feature_range[c][0])
         else:
            normalized_features[r][c] = (all_features[r][c] - feature_range[c][0])

   return normalized_features


##################
##  Best match  ##
##################
def best_match(all_names, all_features, test_image_name, test_image_descriptor):
    rows,columns = all_features.shape

    #loop thru all the test images
    group = {}

    #loop thru all the images
    for j in range(rows):
       s = sum(abs(test_image_descriptor - all_features[j,:]).tolist())
       if test_image_name in group:  # Fixed: replaced has_key() with 'in'
          group[all_names[j]] = s
       else:
          group[all_names[j]] = s

    group = sorted(group.items(), key=lambda x: (x[1], x[0]))  # Fixed: lambda syntax for Python 3

    print("*** Finding best match for image %s ***" % (test_image_name))
    for k in range(NUM_MATCHES):
       if k < len(group):
          print(group[k][0], end=' ')  # Fixed: print with end parameter
    print("\n")

#####################
##  Main Function  ##
#####################
if __name__ == "__main__":
  if len(sys.argv) < 3:
     print("ERROR: python %s file.csv linenum1 [linenum2 ...]" % (sys.argv[0]))
     sys.exit(1)

  #Arguments
  input_csv_file = sys.argv[1]
  
  #Read input file
  all_names, all_features = read_array(input_csv_file)
  print(f"Read {len(all_names)} images with {all_features.shape[1]} features each")  # Debug info

  #normalize data to be between 0 and 1
  frange, normalized_features = normalize_data(all_features)

  #Loop thru the input images
  for i in range(2,len(sys.argv),1):
    line_num = int(sys.argv[i])

    if line_num > len(all_names) - 1:
       print(f"Warning: Line {line_num} exceeds available data (max: {len(all_names)-1})")
       continue

    print(f"Processing line {line_num}: {all_names[line_num]}")  # Debug info
    #Find best match
    best_match(all_names, normalized_features, all_names[line_num], normalized_features[line_num,:])
