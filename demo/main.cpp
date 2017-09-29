#include "demoHeaders.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Invalid arguments" << endl;
        cout << "./main video.mp4" << endl;
        return -1;
    }

    string video_path = argv[1];
    bool is_valid = exists(video_path);
    if (!is_valid)
    {
        cout << "Invalid video file" << argv[1] << endl;        
        return -2;
    }

    try 
    {
        // load video from file
        cv::VideoCapture capVideo = cv::VideoCapture(video_path);
        if (!capVideo.isOpened()) 
        {
             // if not success, exit program
             cout << "Cannot open the video file" << endl;
             return -3;
        }
        
        // Create a dedicated folder
        string basename = boost::filesystem::path(video_path).filename().string();
        size_t lastindex = basename.find_last_of("."); 
        string rawname = basename.substr(0, lastindex); 
        cout << "basename: " << rawname << endl;
        if(!boost::filesystem::create_directory(rawname))
        {
            cout << "Cannot create the directory" << endl;
            return -4;
        }

        // initialize dlib detector and predictor
        frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        cv::Mat frame, resized_frame;

        int number_of_frames = (int)(capVideo.get(CV_CAP_PROP_FRAME_COUNT));
        int fps = (int)(capVideo.get(CV_CAP_PROP_FPS));

        int* xs = new int[N_LANDMARKS];
        int* ys = new int[N_LANDMARKS];
        
        cout << number_of_frames << " frames detected" << endl;
        cout << fps << "fps" << endl;
        cout << capVideo.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << capVideo.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
        int current_frame = 0;
        correlation_tracker tracker; // face tracker
        cout << "Progress:" << endl;
        while (current_frame < number_of_frames)
        {
            cout << "current_frame " << current_frame << endl;
            if (capVideo.read(frame) == false) break; // if not success, break loop

            // here you can use "frame"
            // first of all, resize the frame to a 500-width image
            // work with resized images is faster
            resized_frame = GetSquareImage(frame, 500);

            // turn OpenCV's Mat into something dlib can deal with
            cv_image<bgr_pixel> cimg(resized_frame);
            
            // detect faces
            if (current_frame  == 0){
                // initialize face tracker
                // assumption: in the first frame the face is always available
                std::vector<rectangle> faces = detector(cimg);
                tracker.start_track(cimg, faces[0]);
            } else {
                // update the tracker with a new images
                float confidence = tracker.update(cimg);
            }

            // Get shape of the face from the face-tracker
            rectangle first_face = tracker.get_position();
            
            // extract landmarks
            full_object_detection shape = pose_model(cimg, first_face);

            // print landmarks
            for (int l = 0; l < N_LANDMARKS; ++l)
            {
                int x = shape.part(l).x();
                int y = shape.part(l).y();
                xs[l] = x;
                ys[l] = y;
                //DEBUG: writeCircle(resized_frame, x, y);
            }
            
            // print images
            //DEBUG: cv::imwrite(std::to_string(current_frame) + ".jpg", resized_frame);
                
            string filename = rawname + "/frame" + std::to_string(current_frame) + ".jpg.lm";
            writeCoordinatesOnFile(filename, xs, ys, N_LANDMARKS);
            
            // frame processed - go next
            current_frame = capVideo.get(CV_CAP_PROP_POS_FRAMES);
        }
        
        
        

    }
    catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
    }
}


inline bool exists (const string& name) {
    // Check if a file exists (fastest way)
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

void writeCircle(const cv::Mat& img, int x, int y)
{
    // Draw a circle in a specific point (x, y) in the 'img'
    cv::Point center(x, y);
    cv::circle( img,
            center,
            5,
            cv::Scalar( 0, 0, 255 ));
}

void writeCoordinatesOnFile(string filename, int* xs, int* ys, int length){
    std::ofstream outfile (filename);
    int i = 0;
    for (i = 0; i < length-1; ++i)
    {
        outfile << std::to_string(xs[i]) << " ";
    }
    outfile << std::to_string(xs[length-1]) << std::endl;
    for (i = 0; i < length-1; ++i)
    {
        outfile << std::to_string(ys[i]) << " ";
    }
    outfile << std::to_string(ys[length-1]);
    outfile.close();
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width)
{
    // Scale the images to fit with a max width specified in target_width
    int width = img.cols, height = img.rows;
    cv::Mat dest;

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    float dest_width, dest_height;

    if ( width >= height )
    {
        dest_width = target_width;
        dest_height = height * scale;
    }
    else
    {
        dest_height = target_width;
        dest_width = width * scale;
    }

    cv::resize(img, dest, cv::Size(dest_width, dest_height));
    return dest;
}
