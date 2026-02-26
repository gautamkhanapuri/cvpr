//
// Created by Ajey K on 25/02/26.
//

#include "resnetclassifier.h"

/*
  cv::Mat src        thresholded and cleaned up image in 8UC3 format
  cv::Mat embedding  holds the embedding vector after the function returns
  cv::dnn::Net net   a pre-trained ResNet 18 network (modify the name of the output layer otherwise)
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int ResNetClassifier::getEmbedding(const cv::Mat &src, cv::Mat &embedding, int debug) {
    const int ORNet_size = 224; // expected network input size
    cv::Mat blob;
    cv::Mat resized;

    cv::resize(src, resized, cv::Size(ORNet_size, ORNet_size));

    cv::dnn::blobFromImage(resized, // input image
                           blob, // output array
                           (1.0 / 255.0) * (1 / 0.226), // scale factor
                           cv::Size(ORNet_size, ORNet_size), // resize the image to this
                           cv::Scalar(124, 116, 104), // subtract mean prior to scaling
                           true, // swapRB
                           false, // center crop after scaling short side to size
                           CV_32F); // output depth/type

    this->net.setInput(blob);
    embedding = this->net.forward("onnx_node!resnetv22_flatten0_reshape0");
    if (debug) {
        std::cout << embedding << std::endl;
    }
    return (0);
}

/*
  Given the oriented bounding box information, extracts the region
  from the original image and rotates it so the primary axis is
  pointing right.

  cv::Mat &frame - the original image
  cv::Mat &embimage - the resulting ROI
  int cx - the x coordinate of the centroid of the region
  int cy - the y coordinate of the centroid of the region
  float theta - the orientation of the primary axis of the region (first eigenvector / least 2nd central moment
  float minE1 - the minimum projection value along the primary axis (should be a negative value)
  float maxE1 - the maximum projection value along the primary axis (should be a positive value)
  float minE2 - the minimum projection value along the secondary axis (should be a negative value)
  float maxE2 - the maximum projection value along the secondary axis (should be a positive value)
  int debug - whether to show intermediate images and values

  Note that the expectation is that maxE2 will correspond to up when
  maxE1 is pointing right.  If your system is computing maxE2 as
  pointing down, then swap minE2 and maxE2 and ensure that minE2 is
  negative and maxE2 is positive.
*/
int ResNetClassifier::prepEmbeddingImage(const cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,
                                         float minE1, float maxE1, float minE2, float maxE2, int debug) {
    // rotate the image to align the primary region with the x-axis
    cv::Mat rotatedImage;
    cv::Mat M;

    M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -theta * 180 / M_PI, 1.0);
    int largest = frame.cols > frame.rows ? frame.cols : frame.rows;
    largest = (int) (1.414 * largest);
    cv::warpAffine(frame, rotatedImage, M, cv::Size(largest, largest));
    // cv::warpAffine(frame, rotatedImage, M, frame.size());


    if (debug) {
        cv::imshow("rotated", rotatedImage);
    }

    int left = cx + (int) minE1;
    int top = cy - (int) maxE2;
    int width = (int) maxE1 - (int) minE1;
    int height = (int) maxE2 - (int) minE2;

    // printf("Before bounds check:\n");
    // printf("  cx=%d, cy=%d\n", cx, cy);
    // printf("  minE1=%.1f, maxE1=%.1f, minE2=%.1f, maxE2=%.1f\n",
    //        minE1, maxE1, minE2, maxE2);
    // printf("  Initial: left=%d, top=%d, width=%d, height=%d\n",
    //        left, top, width, height);
    // printf("  rotatedImage size: %dx%d\n", rotatedImage.cols, rotatedImage.rows);


    // bounds check the ROI
    if (left < 0) {
        width += left;
        left = 0;
    }
    if (top < 0) {
        height += top;
        top = 0;
    }
    if (left + width >= rotatedImage.cols) {
        width = (rotatedImage.cols - 1) - left;
    }
    if (top + height >= rotatedImage.rows) {
        height = (rotatedImage.rows - 1) - top;
    }

    if (debug) {
        printf("ROI box: %d %d %d %d\n", left, top, width, height);
    }

    // crop the image to the bounding box of the object
    cv::Rect objroi(left, top, width, height);
    cv::rectangle(rotatedImage, cv::Point2d(objroi.x, objroi.y),
                  cv::Point2d(objroi.x + objroi.width, objroi.y + objroi.height), 200, 4);

    // extract the image and get the embedding of the original image
    cv::Mat extractedImage(rotatedImage, objroi);
    if (debug) {
        cv::imshow("extracted", extractedImage);
    }
    extractedImage.copyTo(embimage);
    return 0;
}

std::vector<float> mat_to_vector(const cv::Mat &embedding) {
    // Embedding is 1x512 or 512x1 Mat
    std::vector<float> vec;

    if (embedding.isContinuous()) {
        // Fast path - data is contiguous in memory
        vec.assign(embedding.ptr<float>(0),
                   embedding.ptr<float>(0) + embedding.total());
    } else {
        // Slow path (shouldn't happen for ResNet output)
        for (int i = 0; i < embedding.total(); i++) {
            vec.push_back(embedding.at<float>(i));
        }
    }

    return vec;
}

cv::Mat vector_to_mat(const std::vector<float> &vec) {
    // Create 1x512 Mat from vector
    cv::Mat embedding(1, vec.size(), CV_32F);

    std::memcpy(embedding.ptr<float>(0), vec.data(), vec.size() * sizeof(float));

    return embedding;
}

int ResNetClassifier::load_data_from_csv() {
    std::vector<char *> temp_labels;
    std::vector<std::vector<float> > temp_features;

    int result = read_image_data_csv(this->db_file_path.string().c_str(),
                                     temp_labels, temp_features, 0);

    if (result != 0) {
        std::cout << "No existing ResNet training data" << std::endl;
        return 0;
    }

    // Convert vector<float> to Mat for each embedding
    for (size_t i = 0; i < temp_labels.size(); i++) {
        this->training_labels.push_back(std::string(temp_labels[i]));

        cv::Mat embedding = vector_to_mat(temp_features[i]);
        this->training_embeddings.push_back(embedding);

        this->starting_datapoint_count++;
    }

    std::cout << "Loaded " << this->starting_datapoint_count
            << " ResNet training examples" << std::endl;

    return 0;
}

int ResNetClassifier::write_trained_data_to_csv() {
    size_t current_size = this->training_labels.size();

    if (current_size <= this->starting_datapoint_count) {
        // Nothing new to write
        return 0;
    }

    // Write only new examples (from starting_datapoint_count onwards)
    for (size_t i = this->starting_datapoint_count; i < current_size; i++) {
        // Convert Mat to vector
        std::vector<float> embedding_vec = mat_to_vector(this->training_embeddings[i]);

        // Append to CSV
        append_image_data_csv(this->db_file_path.string().c_str(),
                              this->training_labels[i].c_str(),
                              embedding_vec);
    }

    std::cout << "Saved " << (current_size - this->starting_datapoint_count)
            << " new ResNet training examples" << std::endl;

    this->starting_datapoint_count = current_size;
    return 0;
}

ResNetClassifier::ResNetClassifier(const std::string &db_file_path) {
    this->resnet_path = resnet_file_path;
    this->db_file_path = db_file_path;
    this->starting_datapoint_count = 0;
    this->training_embeddings.clear();
    this->training_labels.clear();

    this->load_data_from_csv();
}

ResNetClassifier::~ResNetClassifier() {
    std::cout <<
            "ResNetClassifier::~ResNetClassifier appending newly trained data (if trained in this session) to the same csv file: "
            << fs::absolute(this->db_file_path).string() << std::endl;
    this->write_trained_data_to_csv();
}

bool ResNetClassifier::load_resenet18() {
    this->net = cv::dnn::readNetFromONNX(this->resnet_path.string());
    if (this->net.empty()) {
        //    std::cout << "Unable to load network from " << this->resnet_path << std::endl;
        //    std::cout << "ResnetClassifier will not be available during this session" << std::endl;
        return false;
    } else {
        //    std::cout << "Successfully Loaded network from " << this->resnet_path << std::endl;
        //    std::cout << "ResNetClassifier will be available. Press 'r' to toggle resnet classification. Press 'N' to train resnet on images." << std::endl;
        return true;
    }
}

bool ResNetClassifier::has_training_data() {
    return !this->training_labels.empty();
}

bool ResNetClassifier::is_valid_label(std::string &label) {
    return this->labels_set.contains(label);
}

int ResNetClassifier::register_new_label(std::string &label) {
    this->labels_set.insert(label);
    return 0;
}

int ResNetClassifier::train(const cv::Mat &original_frame, const std::vector<RegionStats> &regions) {
    // Preprocess
    std::cout << "Entered training mode for DNN" << std::endl;
    size_t number_of_regions = regions.size();
    std::cout << "Detected " << number_of_regions << " regions." << std::endl;
    std::cout << "Please enter the correct label for the displayed region in lower case." << std::endl;
    std::cout <<
            "The region will be displayed to you. Press 'a' to be able to enter the label in the teminal or press 'p' to skip the region."
            << std::endl;
    std::cout << "If you press 'a', type into the terminal." << std::endl;

    int counter = 1;
    for (const RegionStats &r: regions) {
        cv::Mat embimage;
        // std::cout << "About to train on region " << counter << std::endl;
        // std::cout << "  Extents in train(): minE1=" << r.axisExtent.minE1
        //           << " maxE1=" << r.axisExtent.maxE1
        //           << " minE2=" << r.axisExtent.minE2
        //           << " maxE2=" << r.axisExtent.maxE2 << std::endl;
        this->prepEmbeddingImage(original_frame, embimage,
                                 r.centroid.x, r.centroid.y, r.angle,
                                 r.axisExtent.minE1, r.axisExtent.maxE1,
                                 r.axisExtent.minE2, r.axisExtent.maxE2, 0);

        std::string title = dnn_training_img_display_window_name + " - Region " + std::to_string(counter) + " out of " +
                            std::to_string(number_of_regions);
        cv::imshow(dnn_training_img_display_window_name, embimage);
        cv::setWindowTitle(dnn_training_img_display_window_name, title);
        int key = cv::waitKey(0);
        if (key == 'p') {
            std::cout << "Skipping region" << counter << std::endl;
            continue;
        }
        if (key == 'a') {
            int retry_count = 3;
            std::string user_input_label;
            std::cout << "Available labels:" << std::endl;
            int idx = 1;
            for (const std::string &label: this->labels_set) {
                std::cout << idx << ". " << label << std::endl;
                idx++;
            }
            std::cin >> user_input_label;
            while (!is_valid_label(user_input_label) && retry_count > 0) {
                std::cout << "You typed: '" << user_input_label << "'" << std::endl;
                std::cout << "Is this a new label? <y/n>" << std::endl;
                std::string is_new_label;
                std::cin >> is_new_label;
                if (is_new_label == "y") {
                    this->register_new_label(user_input_label);
                    break;
                }
                std::cout << "Invalid label entered." << std::endl;
                std::cout << "Remaining retries = " << retry_count <<
                        ". Try again. Copy paste the correct label in lower case." << std::endl;
                int idx1 = 1;
                for (const std::string &label: this->labels_set) {
                    std::cout << idx1 << ". " << label << std::endl;
                    idx1++;
                }
                retry_count--;
            }
            if (!is_valid_label(user_input_label) && retry_count == 0) {
                std::cout << "You typed: '" << user_input_label << "'" << std::endl;
                std::cout << "Unrecognized label entered." << std::endl;
                std::cout << "Ignoring this region. Try again later." << std::endl;
                continue;
            }
            std::cout << "Successfully noted down region as: " << user_input_label << std::endl;
            // Get embedding
            cv::Mat embedding;
            this->getEmbedding(embimage, embedding, 0);

            training_embeddings.push_back(embedding.clone());
            training_labels.push_back(user_input_label);

            std::cout << "Trained ResNet on: " << user_input_label << std::endl;
            // this->labels.push_back(user_input_label);
            // this->features.push_back(r.features);
        }
        counter++;
    }
    cv::destroyWindow(dnn_training_img_display_window_name);
    return 0;
}

int ResNetClassifier::classify(const cv::Mat &original_frame, std::vector<RegionStats> &regions) {
    // Same preprocessing
    // std::cout << "Entered classification mode for DNN" << std::endl;
    for (RegionStats &r: regions) {
        cv::Mat embimage;
        // std::cout << "Going to prep embedding image" << std::endl;
        // // Before calling prepEmbeddingImage:
        // std::cout << "Extents: minE1=" << r.axisExtent.minE1
        //           << " maxE1=" << r.axisExtent.maxE1
        //           << " minE2=" << r.axisExtent.minE2
        //           << " maxE2=" << r.axisExtent.maxE2 << std::endl;
        //
        // // Verify:
        // if (r.axisExtent.minE1 > 0 || r.axisExtent.maxE1 < 0 ||
        //     r.axisExtent.minE2 > 0 || r.axisExtent.maxE2 < 0) {
        //     std::cerr << "WARNING: Extent signs are wrong!" << std::endl;
        //     }
        prepEmbeddingImage(original_frame, embimage,
                           r.centroid.x, r.centroid.y, r.angle,
                           r.axisExtent.minE1, r.axisExtent.maxE1,
                           r.axisExtent.minE2, r.axisExtent.maxE2, 0);
        cv::Mat query_embedding;
        getEmbedding(embimage, query_embedding, 0);
        // Find nearest neighbor (SSD)
        double min_dist = std::numeric_limits<double>::max();
        std::string best_label = "Unknown";

        for (size_t i = 0; i < training_embeddings.size(); i++) {
            double dist = cv::norm(query_embedding, training_embeddings[i], cv::NORM_L2SQR);
            if (dist < min_dist) {
                min_dist = dist;
                best_label = training_labels[i];
            }
        }
        if (min_dist < DNN_DISTANCE_THRESHOLD) {
            r.dnn_label = best_label;
        } else {
            r.dnn_label = "Unknown"; // Too far from any training example
        }
    }
    return 0;
}
