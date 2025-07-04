// main.cpp
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

string output_dir = "results";

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
void split_plate(const Mat& plate, Mat& upper, Mat& lower, float upper_ratio = 0.4, float lower_start_ratio = 0.3) {
    int h = plate.rows;
    int upper_y = static_cast<int>(h * upper_ratio);
    int lower_y = static_cast<int>(h * lower_start_ratio);

    upper = plate(Range(0, upper_y), Range::all()).clone();
    lower = plate(Range(lower_y, h), Range::all()).clone();
}

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

int main() {
    if (fs::exists(output_dir)) {
        cout << "📁 기존 출력 폴더 존재함: " << output_dir << " → 삭제 중..." << endl;

        for (const auto& entry : fs::directory_iterator(output_dir)) {
            fs::remove_all(entry);  // 파일 또는 서브디렉토리 전부 제거
        }

        cout << "✅ 폴더 초기화 완료!" << endl;
    } else {
        fs::create_directory(output_dir);
        cout << "📁 출력 폴더 생성됨: " << output_dir << endl;
    }

    string image_path = "/home/Qwd/cplus2/0_1_original.jpg";
    Mat cropped_img = imread(image_path);

    if(cropped_img.empty()) {
        cerr << "이미지를 불러올 수 없습니다!" << endl;
        return -1;
    }
    cout << "===== 번호판 추출 시작 =====" << endl;
    
    // 1. HSV
    Mat hsv;
    cvtColor(cropped_img, hsv, COLOR_BGR2HSV);
    save(hsv, "01_hsv.png");

    // 2. Yellow masking
    Scalar lower_yellow(15, 100, 100);
    Scalar upper_yellow(35, 255, 255);

    Mat mask;
    inRange(hsv, lower_yellow, upper_yellow, mask);
    save(mask, "02_yellow_mask.png");

    Mat masked;
    bitwise_and(cropped_img, cropped_img, masked, mask);
    save(masked, "03_yellow_region.png");

    // 3. 전처리
    Mat gray, blur, edges;
    cvtColor(masked, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);
    Canny(blur, edges, 50, 150);
    save(edges, "04_edges.png");

    // 4. 윤곽선 탐색
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point> candidate;
    
    for(const auto& cnt : contours) {
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.02 * arcLength(cnt, true), true);
        double area = contourArea(approx);

        if (approx.size() == 4 && area > 500) {
            candidate = approx;
            break;
        }
    }
    
    if (!candidate.empty()) {
        // 5. 꼭짓점 찾기
        vector<Point2f> corners = order_points(candidate);

        // 점 찍기
        Mat temp = cropped_img.clone();
        for (const auto& pt : corners) {
            circle(temp, pt, 5, Scalar(0, 255, 0), -1);
        }
        save(temp, "05_detected_corners.png");

        // 6. 번호판 영역 잘라내기
        Point2f tl = corners[0];
        Point2f tr = corners[1];
        Point2f br = corners[2];
        Point2f bl = corners[3];

        int width = static_cast<int>(max(norm(br - bl), norm(tr - tl)));
        int height = static_cast<int>(max(norm(tr - br), norm(tl - bl)));

        vector<Point2f> dst_pts = {
            Point2f(0, 0),
            Point2f(width - 1, 0),
            Point2f(width - 1, height - 1),
            Point2f(0, height - 1)
        };

        // 7. 원근 변환
        Mat M = getPerspectiveTransform(corners, dst_pts);
        Mat warped;
        warpPerspective(cropped_img, warped, M, Size(width, height));
        save(warped, "06_warped_plate.png");

        // 8. 전처리 및 OCR 호출
        // split
        Mat upper, lower;
        split_plate(warped, upper, lower);

        // preprocess
        Mat bin_upper = preprocess(upper);
        Mat bin_lower = preprocess(lower);

        remove_side_dots(bin_upper);

        save(bin_upper, "07_upper.png");
        save(bin_lower, "08_lower.png");

        // ocr
        //string upper_text = ocr_eng(bin_upper, "0123456789");
        string lower_text = ocr_kor(bin_lower, "0123456789가나다라마바사아자하허호거너더러머버서어저고노도로모보소오조구누두루무부수우주배");
        string upper_text = ocr_kor(bin_upper, "0123456789가나다라마바사아자하허호거너더러머버서어저고노도로모보소오조구누두루무부수우주배");

        // clean text
        string cleaned_upper, cleaned_lower;
        for (char c : upper_text) if (isdigit(c)) cleaned_upper += c;
        for (char c : lower_text) if (isdigit(c) || (c & 0x80)) cleaned_lower += c;

        cout << "🔹 상단 인식 결과: " << cleaned_upper << endl;
        cout << "🔸 하단 인식 결과: " << cleaned_lower << endl;
    } else {
        std::cout << "사각형 후보를 찾지 못했습니다!" << std::endl;
    }

    cout << "===== 번호판 추출 완료 =====" << endl;
    return 0;
}
