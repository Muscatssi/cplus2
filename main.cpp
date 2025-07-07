// main.cpp
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

using json = nlohmann::json;
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// 이미지 처리 결과 디렉토리
string output_dir = "results";

//OCR 결과 구조체
struct OcrResult {
    int number;
    int reliability;    // 0: 하단만 정확, 1: 상+하단 정확, -1: 실패
    string lpNum;
    bool success;
};

// 이미지 처리 과정 저장
void save(const Mat& img, const string& filename) {
    string full_path = output_dir + "/" + filename;
    imwrite(full_path , img);
    cout << "이미지 저장 완료: " << full_path  << endl;
}

// 꼭짓점 정렬 함수
vector<Point2f> order_points(const vector<Point>& pts) {
    vector<Point2f> src(4);

    auto sum = [](const Point2f& p) { return p.x + p.y; };
    auto diff = [](const Point2f& p) { return p.y - p.x; };

    src[0] = *min_element(pts.begin(), pts.end(), [&](const Point2f& a, const Point2f& b) { return sum(a) < sum(b); }); // TL
    src[2] = *max_element(pts.begin(), pts.end(), [&](const Point2f& a, const Point2f& b) { return sum(a) < sum(b); }); // BR
    src[1] = *min_element(pts.begin(), pts.end(), [&](const Point2f& a, const Point2f& b) { return diff(a) < diff(b); }); // TR
    src[3] = *max_element(pts.begin(), pts.end(), [&](const Point2f& a, const Point2f& b) { return diff(a) < diff(b); }); // BL

    return src;
}

// 볼트 및 '서울' 제거
void remove_side_dots(Mat& img) {
    int img_width = img.cols;
    int img_height = img.rows;

    // mm → 비율 → 픽셀
    int margin_px = static_cast<int>(img_width * (70.0 / 335.0));  // 약 21% = 0.209
    // 흰색으로 덮기
    rectangle(img, Point(0, 0), Point(margin_px, img_height), Scalar(255, 255, 255), FILLED);
    rectangle(img, Point(img_width - margin_px, 0), Point(img_width, img_height), Scalar(255, 255, 255), FILLED);

    cout << "=== 특정 영역 흰색으로 덮었습니다. ===" << endl;
    save(img, "08_upper_cover.png");
}
// 상/하단 분리
// void split_plate(const Mat& plate, Mat& upper, Mat& lower, float upper_ratio = 0.4, float lower_start_ratio = 0.3) {
//     int h = plate.rows;
//     int upper_y = static_cast<int>(h * upper_ratio);
//     int lower_y = static_cast<int>(h * lower_start_ratio);

//     upper = plate(Range(0, upper_y), Range::all()).clone();
//     lower = plate(Range(lower_y, h), Range::all()).clone();
// }

// OCR 전처리
Mat preprocess(const Mat& input, float resize_factor = 3) {
    Mat gray, resized, binary, kernel;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    resize(gray, resized, Size(), resize_factor, resize_factor, INTER_CUBIC);
    threshold(resized, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    dilate(binary, binary, kernel, Point(-1, -1), 1);
    return binary;
}
// upper img OCR
string ocr_eng(const Mat& image, const string& whitelist) {
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    tess.SetVariable("tessedit_char_whitelist", whitelist.c_str());
    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    tess.SetImage(image.data, image.cols, image.rows, 1, image.step);
    string out = tess.GetUTF8Text();
    tess.End();
    return out;
}
// lower img OCR
string ocr_kor(const Mat& image, const string& whitelist) {
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "kor", tesseract::OEM_LSTM_ONLY);
    tess.SetVariable("tessedit_char_whitelist", whitelist.c_str());
    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    tess.SetImage(image.data, image.cols, image.rows, 1, image.step);
    string out = tess.GetUTF8Text();
    tess.End();
    return out;
}
bool extract_plate_region(const Mat& input, Mat& output_plate, const string& prefix) {
    Mat hsv;
    cvtColor(input, hsv, COLOR_BGR2HSV);
    save(hsv, prefix + "01_hsv.png");

    // 1. 노란색 마스크
    Scalar lower_yellow(15, 100, 100);
    Scalar upper_yellow(35, 255, 255);
    Mat mask;
    inRange(hsv, lower_yellow, upper_yellow, mask);
    save(mask, prefix + "02_yellow_mask.png");

    // 2. 노란 영역 추출
    Mat masked;
    bitwise_and(input, input, masked, mask);
    save(masked, prefix + "03_yellow_region.png");

    // 3. 엣지 추출
    Mat gray, blur, edges;
    cvtColor(masked, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);
    Canny(blur, edges, 50, 150);
    save(edges, prefix + "04_edges.png");

    // 4. 윤곽선 탐색
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Point> candidate;

    for (const auto& cnt : contours) {
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.02 * arcLength(cnt, true), true);
        double area = contourArea(approx);
        if (approx.size() == 4 && area > 500) {
            candidate = approx;
            break;
        }
    }

    if (candidate.empty()) {
        cout << "❌ 번호판 후보 사각형을 찾지 못함" << endl;
        return false;
    }

    // 5. 원근 보정
    vector<Point2f> corners = order_points(candidate);

    // 시각화
    Mat temp = input.clone();
    for (const auto& pt : corners)
        circle(temp, pt, 5, Scalar(0, 255, 0), -1);
    save(temp, prefix + "05_detected_corners.png");

    Point2f tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];
    int width = static_cast<int>(max(norm(br - bl), norm(tr - tl)));
    int height = static_cast<int>(max(norm(tr - br), norm(tl - bl)));

    float fx = static_cast<float>(width - 1);
    float fy = static_cast<float>(height - 1);

    vector<Point2f> dst_pts = {
        {0.0f, 0.0f},
        {fx, 0.0f},
        {fx, fy},
        {0.0f, fy}
    };

    Mat M = getPerspectiveTransform(corners, dst_pts);
    warpPerspective(input, output_plate, M, Size(width, height));
    save(output_plate, prefix + "06_warped_plate.png");

    return true;
}

OcrResult process_plate(const Mat& input_img, int index) {
    OcrResult result;
    result.number = index;
    result.reliability = -1;
    result.lpNum = "";
    result.success = false;

    Mat plate_img;
    if (!extract_plate_region(input_img, plate_img, "img_" + to_string(index) + "_")) {
        // 번호판 검출 실패
        cout << "❌ [" << index << "] 번호판 사각형 추출 실패" << endl;
        return result;  //success = false 유지
    }

    // 1. 상/하단 분리
    Mat upper, lower;
    int h = plate_img.rows;
    upper = plate_img(Range(0, int(h * 0.4)), Range::all()).clone();
    lower = plate_img(Range(int(h * 0.3), h), Range::all()).clone();

    // 2. ocr 전처리
    Mat bin_upper = preprocess(upper);
    Mat bin_lower = preprocess(lower);

    remove_side_dots(bin_upper);

    // 3. OCR (Tesseract 호출)
    string upper_text = ocr_eng(bin_upper, "0123456789");
    string lower_text = ocr_kor(bin_lower, "0123456789가나다라마바사아자하허호");

    // 4. 텍스트 정리
    string cleaned_upper, cleaned_lower;
    for (char c : upper_text) if (isdigit(c)) cleaned_upper += c;
    for (char c : lower_text) if (isdigit(c) || (c & 0x80)) cleaned_lower += c;

    // 5. 판단
    result.reliability = (!cleaned_upper.empty() && !cleaned_lower.empty()) ? 1 :
                         (!cleaned_lower.empty()) ? 0 : -1;
    result.lpNum = cleaned_upper + " " + cleaned_lower;
    result.success = (result.reliability != -1);

    return result;
}


int main() {
    vector<OcrResult> results;
    int pass = 0, fail = 0;
    int index = 0;

    if (!fs::exists(output_dir)) {
        cout << "📂 result 폴더 생성 중 ..." << endl;
        fs::create_directory(output_dir);
        cout << "📂 result 폴더 생성 완료" << endl;
    } else {
        cout << "📂 result 폴더 이미 존재합니다." << endl;
    }

    for (const auto& entry : fs::directory_iterator("images")) {
        if (!entry.is_regular_file()) continue;

        Mat img = imread(entry.path().string());
        if (img.empty()) {
            cerr << "❌ 이미지 로딩 실패: " << entry.path() << endl;
            continue;
        }

        OcrResult res = process_plate(img, index);
        results.push_back(res);

        if (res.success) pass++;
        else fail++;

        index++;
    }
    json j;
    j["pass"] = pass;
    j["fail"] = fail;
    j["buses"] = json::array();

    for (const auto& r : results) {
        j["buses"].push_back({
            {"number", r.number},
            {"reliability", r.reliability},
            {"lpNum", r.lpNum}
        });
    }

    ofstream file(output_dir + "/result.json");
    file << j.dump(4);
    file.close();

    cout << "✅ OCR JSON 저장 완료: " << output_dir + "/result.json" << endl;

    return 0;
}