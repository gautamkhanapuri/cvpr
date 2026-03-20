//
// Created by Ajey K on 15/03/26.
//

#include "caar.h"

CAAR::CAAR(fs::path& calibration_file_path): calib(calibration_file_path), ar() {
    std::cout << "Initializing CAAR (Calibration and Augmented Reality) object." << std::endl;
    this->vd_cap = nullptr;
    this->refS = cv::Size(0, 0);
    this->mode = 1;
    this->display_mode_switch_notification = false;

    this->vid_setup();
    this->calib.read_calibration_data(this->refS);
}

CAAR::~CAAR() {
    std::cout << "Destroying CAAR..." << std::endl;
    delete this->vd_cap;
    this->calib.write_calibration_data();
    // std::cout << "Updated Calibration matrix written to: " << fs::absolute(this->calibration_file_path)
    //         << std::endl;
    // std::cout << "Calibration matrix was over written if it contained data earlier." << std::endl;
}

int CAAR::vid_setup() {
    std::cout << "Starting up video device..." << std::endl;
    this->vd_cap = new cv::VideoCapture(device_id, api_id);
    // std::cout << "Successfully accessed device: " << this->vd_cap->getBackendName() << std::endl;


    // Failure case of video capture not starting
    if (!this->vd_cap->isOpened()) {
        std::cerr << "Unable to access video device." << std::endl;
        std::cerr << "Terminating..." << std::endl;
        std::exit(-1);
    }

    this->refS = cv::Size(static_cast<int>(this->vd_cap->get(cv::CAP_PROP_FRAME_WIDTH)),
                          static_cast<int>(this->vd_cap->get(cv::CAP_PROP_FRAME_HEIGHT)));
    std::cout << "Video Display Size - \nWidth: " << refS.width << "\nHeight: " << refS.height << std::endl;
    return 0;
}

int CAAR::run() {
    cv::namedWindow(window1, 1);
    std::cout << "Mode starts up in: " << mode_number_to_name_map.at(this->mode) << std::endl;
    std::cout << "Displaying Real Time video now..." << std::endl;

    for (;;) {
        *this->vd_cap >> main_frame;
        if (main_frame.empty()) {
            std::cerr << "No video frame." << std::endl;
            std::cerr << "Terminating..." << std::endl;
            std::exit(-1);
        }

        if (this->mode == 1) {
            // std::cout << "Inside mode 1" << std::endl;
            this->calib.detect_and_mark_corners(this->main_frame);
        } else if (this->mode == 2) {
            // this->display_frame = this->main_frame.clone();
            this->ar.display_objects(this->main_frame);
        }

        if (display_mode_switch_notification) {
            this->display_notifications();
        }

        cv::imshow(window1, this->main_frame);

        this->pressed = cv::waitKey(10);
        this->handle_key();

        if (this->pressed == 'q') {
            std::cout << "Received 'q'. Exiting video loop..." << std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}

int CAAR::handle_key() {
    if (mode == 1) {
        this->handle_key_in_calibration_mode();
    }
    else if (mode == 2) {
        this->handle_key_in_reprojection_mode();
    }
    return 0;
}

int CAAR::handle_key_in_calibration_mode() {
    switch (this->pressed) {
        case 'r':
            if (this->calib.is_calibration_matrix_available()) {
                this->mode = 2;
                this->ar.set_calibration_matrix(this->calib.get_calibration_matrix());
                this->ar.set_distortion_matrix(this->calib.get_distortion_matrix());
            } else {
                this->display_mode_switch_notification = true;
                this->mode_switch_notification_countdown_timer = timer_mode_switch_notification;
            }
            break;

        case 's': {
                this->calib.save_calibration_image();
                break;
            }

        default:
            break;
    }
    return 0;
}

int CAAR::handle_key_in_reprojection_mode() {
    switch (this->pressed) {
        case 'c':
            this->mode = 1;
            break;

        case 's': {
            const long long time_now = get_time_instant();
            const std::string time_now_str = std::to_string(time_now);

            const std::string ar_image_path = ar_image_filename + time_now_str + image_save_format;
            cv::imwrite(ar_image_path, this->main_frame);
            std::cout << "Saved AR image: " << ar_image_path << std::endl;
            break;
        }

        case '0':
            this->ar.show_axes();
            break;

        case '1':
            this->ar.ar_object1();
            break;

        case '2':
            this->ar.ar_object2();
            break;

        case '3':
            this->ar.ar_object3();
            break;

        default:
            break;
    }
    return 0;
}

int CAAR::display_notifications() {
    int fontFace = cv::QT_FONT_NORMAL;
    double fontScale = 0.6; // Small but legible
    int thickness = 1;
    cv::Scalar color(0, 0, 255); // Red in BGR format

   int i = 1;
    if (this->display_mode_switch_notification) {
        int baseline = 0;
        std::string text = std::format("Reprojection mode UNAVAILABLE. Need at least {} calibration images. {} more images required.", min_calibration_images_required, min_calibration_images_required - this->calib.get_calibration_images_count());
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg((this->main_frame.cols - textSize.width) / 2, textSize.height + (i * 20));
        i++;
        cv::putText(this->main_frame, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);
        this->mode_switch_notification_countdown_timer--;
        this->display_mode_switch_notification = this->mode_switch_notification_countdown_timer < 0;
    }
    return 0;
}
