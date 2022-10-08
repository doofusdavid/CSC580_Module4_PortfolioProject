"""
Module 4 - Portfolio Milestone - Option 1
David Edwards
CSC580: Applying Machine Learning and Neural Networks - Capstone
Colorado State University-Global Campus
Dr. Joseph Issa
October 9, 2022
"""

import PIL.Image
import PIL.ImageDraw
import face_recognition
import sys
import os 

# Main function
def main(argv):
    # Make sure the user provided the correct files
    if len(argv) != 2:
        print("Usage: Edwards_David.py <checkPersonImage> <checkGroupImage>")
        print("Example: Edwards_David.py img/person1.jpg img/group1.jpg")
        print("Program attempts to find the person in the checkPersonImage in the checkGroupImage, and will display the checkGroupImage with a box around the person's face if found.")
        return
    if not os.path.isfile(argv[0]) or not os.path.isfile(argv[1]):
        print("Error: one or both files do not exist")
        return
    
    person_filename = argv[0]
    group_filename = argv[1]
    

    person_image = face_recognition.load_image_file(person_filename)
    group_image = face_recognition.load_image_file(group_filename)
    
    person_face_encoding = face_recognition.face_encodings(person_image)[0]

    # make sure to get the face locations so we can draw around them if found
    group_face_locations = face_recognition.face_locations(group_image)
    group_face_encodings = face_recognition.face_encodings(group_image)

    # compare the faces to see if there are matching faces
    
    results = face_recognition.compare_faces(group_face_encodings, person_face_encoding)
    
    for i, is_match in enumerate(results):
        if is_match:
            print("Found a match in the group file!")
            print("Displaying highlighted image")
            top, right, bottom, left = group_face_locations[i]
            pil_image = PIL.Image.fromarray(group_image)
            draw = PIL.ImageDraw.Draw(pil_image)
            draw.rectangle(((left, top), (right, bottom)), outline=(148,0,211), width=3)
            pil_image.show()
            break
    else:
        print("No match found")


if __name__ == "__main__":
   main(sys.argv[1:])