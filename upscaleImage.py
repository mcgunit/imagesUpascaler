import cv2, pyexiv2, os, argparse
from cv2 import dnn_superres




if __name__ == "__main__":


    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--inputFolder", help="absolute path to folder of the images to scale")
    argParser.add_argument("-o", "--outputFolder", help="absolute path to folder to store scaled images")

    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.inputFolder=%s" % args.inputFolder)
    print("args.outputFolder=%s" % args.outputFolder)

    dirPath = args.inputFolder
    dirOut = args.outputFolder

    try: 
        # Create an SR object for image upscaling
        sr = dnn_superres.DnnSuperResImpl_create()
        #print("Current working dir", os.getcwd())
        modelsPath = os.path.join(os.getcwd(), "models")
        path = os.path.join(modelsPath, "EDSR_x4.pb")
        sr.readModel(path)
        sr.setModel("edsr", 4)
        #sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except:
        print("Failed to load super resolution model")

    try:
        if not os.path.exists(dirOut):
            os.makedirs(dirOut)

        for image in os.listdir(dirPath):
            if image.endswith(('jpg', 'png', 'jpeg', 'JPG')):
                # read image from given path
                dis = cv2.imread(os.path.join(dirPath, image), 1)
                
                # read with exif
                originalImage = pyexiv2.Image(os.path.join(dirPath, image))
                try:
                    originalExif = originalImage.read_exif()
                except:
                    originalExif = None
                try: 
                    originalXMP = originalImage.read_xmp()
                except:
                    originalXMP = None

                originalImage.close()

                # Upscale the image
                dis = sr.upsample(dis)
            
            
                # Save the image with exif
                cv2.imwrite(os.path.join(dirOut, image), dis)
                newImage = pyexiv2.Image(os.path.join(dirOut, image))
                if not originalExif == None:
                    try:
                        newImage.modify_exif(originalExif)
                    except:
                        print("Failed to write exif tags")
                if not originalXMP == None:
                    try:
                        newImage.modify_xmp(originalXMP)
                    except:
                        print("Failed to write xmp tags")
                newImage.close()

    except Exception as e:
        print("Failed to upscale image: ", e)
