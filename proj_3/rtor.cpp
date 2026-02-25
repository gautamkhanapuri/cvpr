//
// Created by Ajey K on 21/02/26.
//

#include "rtor.h"

#include "utils.h"

RTObectRecognizer::RTObectRecognizer(const fs::path& db_filepath): db_filepath(db_filepath), threshold(threshold_mode), classifier(db_filepath) {
    std::cout << "Initialised RTObectRecognizer..." << std::endl;
    this->vd_cap = nullptr;
    this->device_id = 0;
    this->api_id = cv::CAP_AVFOUNDATION;
    this->show_binary = false;
    this->show_morphology = false;

    vid_setup();
}

RTObectRecognizer::~RTObectRecognizer() {
    std::cout << "Destroying RTObectRecognizer..." << std::endl;
    delete this->vd_cap;
    this->classifier.write_new_trained_data();
    std::cout << "All new data trained in this session appended to the same file: " << fs::absolute(this->db_filepath) << std::endl;
}

int RTObectRecognizer::vid_setup() {
    std::cout << "Starting up video device..." << std::endl;
    this->vd_cap = new cv::VideoCapture(this->device_id, cv::CAP_AVFOUNDATION);
    std::cout << "Successfully opened device: " << this->vd_cap->getBackendName() << std::endl;


    // Failure case of video capture not starting
    if (!this->vd_cap->isOpened()) {
        std::cerr << "Unable to open video device." << std::endl;
        std::cerr << "Terminating..." << std::endl;
        std::exit(-1);
    }

    this->refS = cv::Size(static_cast<int>(this->vd_cap->get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(this->vd_cap->get(cv::CAP_PROP_FRAME_HEIGHT)));
    std::cout << "Video Display Initialized.\nWidth: " << refS.width << "\nHeight: " << refS.height << std::endl;
    return 0;
}

int RTObectRecognizer::run() {
    cv::namedWindow(window1, 1);
    std::cout << "Displaying Real Time video now..." << std::endl;

    for (;;) {
        *this->vd_cap >> main_frame;
        if (main_frame.empty()) {
            std::cerr << "No video frame." << std::endl;
            std::cerr << "Terminating..." << std::endl;
            std::exit(-1);
        }
        // cv::imwrite("test.jpg", main_frame);

        if (threshold_mode == 1  && !this->white_screen_set) {
            std::cout << "Reading white screen. Ensure platform is empty with only white background." << std::endl;
            std::cout << "White screen can be reset later by pressing the key 'w'. " << std::endl;
            bool white_screen_is_clean = this->threshold.pickup_white_screen(this->main_frame);
            if (!white_screen_is_clean) {
                std::cout << "White screen not picked up. Clean WS." << std::endl;
                cv::imshow(window1, this->main_frame);
                this->pressed = cv::waitKey(10);
                continue;
            }
            this->white_screen_set = true;
            std::cout << "White screen captured! Starting object recognition..." << std::endl;
        }

        this->threshold.threshold(this->main_frame, this->bin_frame);
        morph(this->bin_frame, this->morph_frame);

        std::vector<RegionStats>  region_stats;
        this->segment.make_segments(this->morph_frame, this->label_map, region_stats);
        this->feature.calculate_basic_2d_features(region_stats);

        if (this->classifier.has_training_data()) {
            this->classifier.predict(region_stats);
        }

        this->display_frame = this->main_frame.clone();
        this->tracker.match_and_draw_box(this->display_frame, region_stats);
        this->feature.overlay_features(this->display_frame, region_stats);

        cv::imshow(window1, this->display_frame);

        if (show_binary) {
            cv::imshow(threshold_binary_window, this->bin_frame);
        }

        if (show_morphology) {
            cv::imshow(cleaned_morph_window, this->morph_frame);
        }

        this->pressed = cv::waitKey(10);
        this->handle_key(this->pressed, region_stats);

        if (this->pressed == 'q') {
            std::cout << "Terminating program..." << std::endl;
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

int RTObectRecognizer::handle_key(int key, std::vector<RegionStats> &regions) {
    switch (key) {
        case 'n':
            this->classifier.train_on_all_segments(this->main_frame, this->label_map, regions);
            break;

        case 't':
            this->show_binary = !this->show_binary;
            if (!this->show_binary) {
                cv::destroyWindow(threshold_binary_window);
            }
            break;

        case 'm':
            this->show_morphology = !this->show_morphology;
            if (!this->show_morphology) {
                cv::destroyWindow(cleaned_morph_window);
            }
            break;

        case 'w':
            if (threshold_mode == 1) {
                this->white_screen_set = false;
            }
            break;

        case 's': {
            const long long time_now = get_time_instant();
            const std::string time_now_str = std::to_string(time_now);

            const std::string original_image_path = main_image_filename + time_now_str + image_save_format;
            cv::imwrite(original_image_path, this->main_frame);

            const std::string overlay_image_path = overlay_image_filename + time_now_str + image_save_format;
            cv::imwrite(overlay_image_path, this->display_frame);

            const std::string threshold_image_path = threshold_image_filename + time_now_str + image_save_format;
            cv::imwrite(threshold_image_path, this->bin_frame);

            const std::string morphed_image_path = morphed_image_filename + time_now_str + image_save_format;
            cv::imwrite(morphed_image_path, this->morph_frame);

            break;
        }

        default: ;
    }
    return 0;
}


