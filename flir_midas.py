import cv2
import PySpin
import matplotlib.pyplot as plt

def main():
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()


     # Run example on each camera
    for i, cam in enumerate(system.GetCameras()):

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        cam.BeginAcquisition()

        # Execute software trigger
        node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
        if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
            print('Unable to execute trigger. Aborting...')
            return False

        while(True):

            node_softwaretrigger_cmd.Execute()

            #  Retrieve next received image
            image_result = cam.GetNextImage(1000)

            if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                
                # Convert image to Mono8
                # image_converted = image_result.Convert(PySpin.PixelFormat_BayerRG8, PySpin.HQ_LINEAR)

                # get image data
                image = image_result.GetNDArray()

                image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB_EA)

                # Display image
                plt.imshow(image, cmap='gray')
                plt.show()
                
        # End acquisition
        cam.EndAcquisition()

if __name__ == '__main__':
    main()