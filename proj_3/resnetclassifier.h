//
// Created by Ajey K on 25/02/26.
// Header file for ResNet18 embedding-based one-shot classification.
// Implements deep learning embeddings for object recognition using pre-trained ResNet18 network.
//

#ifndef RESNETCLASSIFIER_H
#define RESNETCLASSIFIER_H

#include <cstdio>
#include <cstring>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

#include "segment.h"
#include "csv_util.h"

namespace fs = std::filesystem;

// Path to ResNet18 ONNX model file (must be in same directory as executable)
inline const std::string resnet_file_path = "resnet18-v2-7.onnx";

// Window title for ResNet training mode display
inline const std::string dnn_training_img_display_window_name = "ResNet18 image training";

// Distance threshold for ResNet classification (L2 squared distance in embedding space)
inline const double DNN_DISTANCE_THRESHOLD = 200.0;

/**
 * Converts OpenCV Mat (1x512 or 512x1) to vector of floats for CSV storage.
 * @param embedding ResNet output embedding matrix
 * @return vector of 512 float values
 */
std::vector<float> mat_to_vector(const cv::Mat &embedding);

/**
 * Converts vector of floats to OpenCV Mat for embedding comparison.
 * @param vec vector of embedding values from CSV
 * @return Mat in 1x512 format
 */
cv::Mat vector_to_mat(const std::vector<float> &vec);

/**
 * One-shot classifier using ResNet18 embeddings.
 * Requires only single training example per category for classification.
 */
class ResNetClassifier {
    cv::dnn::Net net; // ResNet18 DNN model
    fs::path resnet_path; // Path to ONNX model file
    fs::path db_file_path; // Path to embeddings database CSV
    int starting_datapoint_count; // Number of examples loaded from file
    std::vector<cv::Mat> training_embeddings; // 512D embedding vectors for training examples
    std::vector<std::string> training_labels; // Labels for training examples
    std::set<std::string> labels_set; // Set of unique labels

    /**
     * Extracts 512-dimensional embedding from preprocessed 224x224 image.
     * Uses ResNet18 global average pooling layer output.
     * @param src preprocessed image (224x224)
     * @param embedding output 512D vector
     * @param debug 1 to print embedding values, 0 for silent
     * @return 0 if successful
     */
    int getEmbedding(const cv::Mat &src, cv::Mat &embedding, int debug);

    /**
     * Preprocesses object for ResNet input following standard pipeline:
     * Rotates image to align principal axis, extracts ROI, resizes to 224x224.
     * @param frame original color image
     * @param embimage output preprocessed image
     * @param cx centroid x-coordinate
     * @param cy centroid y-coordinate
     * @param theta principal axis angle in radians
     * @param minE1 minimum extent along major axis (negative)
     * @param maxE1 maximum extent along major axis (positive)
     * @param minE2 minimum extent along minor axis (negative)
     * @param maxE2 maximum extent along minor axis (positive)
     * @param debug 1 to show intermediate images, 0 for silent
     * @return 0 if successful
     */
    int prepEmbeddingImage(const cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta, float minE1,
                           float maxE1, float minE2, float maxE2, int debug);

    /**
     * Loads training embeddings and labels from CSV file.
     * @return 0 if successful
     */
    int load_data_from_csv();

    /**
     * Appends newly collected training examples to CSV file.
     * @return 0 if successful
     */
    int write_trained_data_to_csv();

    /**
     * Checks if label exists in current label set.
     * @param label label to validate
     * @return true if label exists
     */
    bool is_valid_label(std::string &label);

    /**
     * Registers new label category.
     * @param label new label to add
     * @return 0 if successful
     */
    int register_new_label(std::string &label);

public:
    /**
     * Constructor. Loads existing embedding database and ResNet model.
     * @param db_file_path path to embeddings CSV file
     */
    ResNetClassifier(const std::string &db_file_path);

    /**
     * Destructor. Saves newly collected training data to CSV.
     */
    ~ResNetClassifier();

    /**
     * Loads ResNet18 ONNX model using OpenCV DNN module.
     * @return true if model loaded successfully, false otherwise
     */
    bool load_resenet18();

    /**
     * Checks if training database has examples.
     * @return true if training data exists
     */
    bool has_training_data();

    /**
     * Interactive training mode for ResNet embeddings.
     * Displays preprocessed objects and collects labels.
     * @param original_frame original color frame
     * @param regions detected regions with computed extents
     * @return 0 if successful
     */
    int train(const cv::Mat &original_frame, const std::vector<RegionStats> &regions);

    /**
     * Classifies regions using nearest-neighbor in ResNet embedding space.
     * @param original_frame original color frame
     * @param regions detected regions (dnn_label assigned to each)
     * @return 0 if successful
     */
    int classify(const cv::Mat &original_frame, std::vector<RegionStats> &regions);
};

#endif //RESNETCLASSIFIER_H
