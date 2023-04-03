######### PART 1 - Importing Libraries
import cv2 
import easyocr as ocr 
import numpy as np 
from PIL import Image
import pytesseract as pt
from pytesseract import Output
import streamlit as st 

######### PART 2 - APPLICATION FRONT-END UPPER SECTION
# Title
st.title("Ai Recognition & Image Processing Software")

# Subtitle
st.markdown("### Computer Vision & Machine Learning Application")
st.markdown("")

# Radio Buttons
genre = st.radio("The first 4 options of this application read/recognize characters within a single image and extract its " + 
                 "readable content and convert it into regular text (scroll towards the bottom of " + 
                 "the page after uploading your file). The 5th option modifies the colors of you pictures using " + 
                 "complex mathematical morphology. Printed or typed text is selected by default. " + 
                 "Make sure your image has the right position (not flipped upside down / sideways) nor " +
                 "previously modified or altered. " +
                 "Please select the correct option for your PDF before dragging and dropping your file: ",
    ('Typed Text - Document', 
     'Signs - Outdoors', 
     'Invoice - Paper Store Receipts', 
     'Official ID - Passport or License',
     'Photo - Ai Cartoonizer'), index=0)

# Image Uploader
image = st.file_uploader(label = "Upload your PDF File here", type = ['jpeg','jpg','png'])

# st. cache_resource is the recommended way to cache global resources like ML models or db connections
#  – unserializable objects that you don't want to load multiple times.
@st.cache_resource

# Define model for EasyOCR
def load_model_1(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader 
# Load EasyOCR Model
reader_1 = load_model_1()

######### PART 3 - APPLICATION BACK-END / AI / MACHINE LEARNING / IMAGE PROCESSING

# Ai gets applied once an image gets uploaded
if image is not None:
    #### A. APPLY EASYOCR MODEL WHEN USER SELECTS 'DOCUMENT'
    if genre == 'Typed Text - Document':
        try:
            # Read image
            input_image = Image.open(image)
            # Display Image
            st.image(input_image) #display image

            with st.spinner("Ai will compile your results shortly..."): 
                # reader_1 creates probablity based on character and selects a label   
                result_1 = reader_1.readtext(np.array(input_image))
                # Empty list created for appended text labels
                result_text_1 = []
                for text in result_1:
                    result_text_1.append(text[1])
                # Results get displayed at front-end
                st.success("Text Data is Ready")
                st.markdown("#### TEXT RESULTS (please verify results as typos may occur):")
                st.write(' '.join(map(str, result_text_1)))
        except:
            st.markdown("Error - your image doesn't contain any printed text nor any characters at all. " + 
                        "If that's not the case - try an image with a better resolution or perhaps " + 
                        "you selected the wrong model.")
    #### B. APPLY EASYOCR MODEL WHEN USER SELECTS 'SIGNS'
    elif genre == 'Signs - Outdoors':
        try:
            input_image = Image.open(image) #read image
            with st.spinner("Ai will compile your results shortly..."): 
                # reader_1 creates probablity based on character and selects a label  
                result_1 = reader_1.readtext(np.array(input_image))
                # Empty list created for appended text labels
                result_text_1 = []
                for text in result_1:
                    result_text_1.append(text[1])

                # Set up 'boxes' for detected text
                top_left = tuple(result_1[0][0][0])
                bottom_right = tuple(result_1[0][0][2])
                text = result_1[0][1]
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Open image once again for assigning 'boxes' into characters
                image_v = Image.open(image)
                frame = np.array(image_v)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                spacer = 100
                for detection in result_1: 
                    top_left = tuple(detection[0][0])
                    bottom_right = tuple(detection[0][2])
                    text = detection[1]
                    frame = cv2.rectangle(frame, top_left,bottom_right,(0,255,0),3)
                    frame = cv2.putText(frame, text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                    spacer+=15

                # Results get displayed at front-end     
                st.image(frame)       
                st.success("Text Data is Ready")
                st.markdown("#### TEXT RESULTS (please verify results as typos may occur):")
                st.write(' '.join(map(str, result_text_1)))    
        except:
            st.markdown("Error - your image doesn't contain any printed text nor any characters at all. " + 
                        "If that's not the case - try an image with a better resolution or perhaps " + 
                        "you selected the wrong model.")
    #### C. APPLY PYTESSERACT MODEL WHEN USER SELECTS 'INVOICE'
    elif genre == 'Invoice - Paper Store Receipts':
        try:
            with st.spinner("Ai will compile your results shortly..."):
                # Convert image into file bytes for data to become available for cv2
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
        
                ## PRE-PROCESSING
                image=cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
                se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
                bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
                out_gray=cv2.divide(image, bg, scale=255)
                out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
                ret,thresh1 = cv2.threshold(out_binary, 210, 255, cv2.THRESH_BINARY)
                d = pt.image_to_data(opencv_image, output_type=Output.DICT)
                n_boxes = len(d['level'])
                boxes = cv2.cvtColor(opencv_image.copy(), cv2.COLOR_BGR2RGB)
                for i in range(n_boxes):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
                    boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = pt.image_to_string(opencv_image, lang = 'eng')

                # Results get displayed at front-end     
                st.image(boxes)       
                st.success("Text Data is Ready")
                st.markdown("#### TEXT RESULTS (please verify results as typos may occur):")
                st.write(text)
        except:     
            st.markdown("Error - your image doesn't contain any text from a printed receipt nor any characters at all. " + 
                        "If that's not the case - try an image with a better resolution or perhaps " + 
                        "you selected the wrong model.")
    #### D. PYTESSERACT MODELS WHEN USER SELECTS 'ID'       
    elif genre == 'Official ID - Passport or License':
        try:
            #input_image = Image.open(image) #read image
            with st.spinner("Ai will compile your results shortly..."):
                # Convert image into file bytes for data to become available for cv2
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)

                ## PRE-PROCESSING
                image=cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
                se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
                bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
                out_gray=cv2.divide(image, bg, scale=255)
                out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]    
                d = pt.image_to_data(out_binary, output_type=Output.DICT)
                n_boxes = len(d['level'])
                boxes = cv2.cvtColor(out_binary.copy(), cv2.COLOR_BGR2RGB)
                for i in range(n_boxes):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
                    boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = pt.image_to_string(out_binary, lang = 'eng')

                # Results get displayed at front-end     
                st.image(boxes)       
                st.success("Text Data is Ready")
                st.markdown("#### TEXT RESULTS (please verify results as typos may occur):")
                st.write(text)
        except:
            st.markdown("Error - your image doesn't contain any official passport nor identification at all. " + 
                        "If that's not the case - try an image with a better resolution.")
    #### E. APPLY IMAGE PROCESSING TECHNIQUES WHEN USER SELECTS 'CARTOONIZER'             
    elif genre == 'Photo - Ai Cartoonizer':
        try:
            with st.spinner("Ai will compile your picture shortly..."):
                # Convert image into file bytes for data to become available for cv2
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)

                # Modify Colors
                img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                gsimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rs2 = cv2.resize(gsimg, (960, 540))
                sgscl = cv2.medianBlur(gsimg, 5)
                rs3 = cv2.resize(sgscl, (960, 540)) 

                # Black/White Edges 
                ge = cv2.adaptiveThreshold(sgscl, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                rs4 = cv2.resize(ge, (960, 540))

                # Cartoon-like colors
                cimg = cv2.bilateralFilter(img, 9, 300, 300)
                rs5 = cv2.resize(cimg, (960, 540))
                ci = cv2.bitwise_and(cimg, cimg, mask=ge)
                rs6 = cv2.resize(ci, (960, 960))

                # Results get displayed at front-end           
                st.success("Your Pictures Are Ready - Enjoy!")
                st.markdown("## Photo 1")
                st.image(rs4) 
                st.markdown("## Photo 2")
                st.image(rs6)   
        except:
            st.markdown("Error - Try again with another picture.")         
    else:
        st.write("Disclaimer: Results will vary depending on your selection, and the quality of your image and/or text's appearance.")

# End of IF-LOOP
######### PART 4 - APPLICATION FRONT-END BOTTOM SECTION
st.markdown("")
st.caption("Computer Vision / Machine Learning Product Made by Juan L. Moctezuma-Flores")

################################### END OF APPLICATION ###################################

