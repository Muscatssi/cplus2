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
#include <regex>

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
    rectangle(img, Point(0, 0), Point(img_width / 2, img_height), Scalar(255, 255, 255), FILLED);
    rectangle(img, Point(img_width - margin_px, 0), Point(img_width, img_height), Scalar(255, 255, 255), FILLED);

    cout << "=== 특정 영역 흰색으로 덮었습니다. ===" << endl;
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

// 정규식 검증용
bool validate_lower_pattern(const string& str) {
    // 1. UTF-8 문자열을 각 문자를 제대로 다룰 수 있는 wstring으로 변환
    wstring_convert<codecvt_utf8<wchar_t>> conv;
    wstring wstr = conv.from_bytes(str);

    // 2. 5글자인지 확인
    if (wstr.length() != 5) return false;

    // 3. 첫 번째 글자가 한글인지 확인
    if (!(wstr[0] >= L'가' && wstr[0] <= L'힣')) return false;
    
    // 4. 나머지 네 글자가 모두 숫자인지 확인
    for (int i = 1; i < 5; ++i) if (!iswdigit(wstr[i])) return false;
    return true;
}

// OCR 전처리
Mat preprocess(const Mat& input, float resize_factor = 2.7) {
    Mat gray, resized, binary, kernel;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    resize(gray, resized, Size(), resize_factor, resize_factor, INTER_CUBIC);
    
    kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat background;
    morphologyEx(resized, background, MORPH_CLOSE, kernel);
    Mat diff;
    absdiff(resized, background, diff);
    bitwise_not(diff, diff);

    threshold(diff, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    dilate(binary, binary, kernel, Point(-1, -1), 1);
    return binary;
}

bool extract_plate_region(const Mat& input, Mat& output_plate, const string& prefix) {
    Mat hsv;
    cvtColor(input, hsv, COLOR_BGR2HSV);
    save(hsv, prefix + "01_hsv.png");

    // // 1. 노란색 마스크
    // Scalar lower_yellow(15, 100, 100);
    // Scalar upper_yellow(35, 255, 255);
    // 확장된 범위 (어두운 노랑 허용)
    Scalar lower_yellow(10,  50,  50);  // Hue 약간 낮추고, Saturation/Value도 완화
    Scalar upper_yellow(40, 255, 255);

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
    save(bin_upper, "img_" + to_string(index) + "_" + "07_upper_image.png");
    save(bin_lower, "img_" + to_string(index) + "_" + "07_lower_image.png");

    // 3. OCR (Tesseract 호출)
    string upper_text = ocr_kor(bin_upper, "0123456789");
    string lower_text = ocr_kor(bin_lower, "0123456789가나다라마바사아자하허호");

    // 4. 텍스트 정리
    upper_text.erase(remove(upper_text.begin(), upper_text.end(), '\n'), upper_text.end());
    lower_text.erase(remove(lower_text.begin(), lower_text.end(), '\n'), lower_text.end());
    
    string cleaned_upper, cleaned_lower;
    
    for (char c : upper_text) if (isdigit(c)) cleaned_upper += c;
    cleaned_lower = lower_text;

    cout << "[" << index << "] upper_text: " << upper_text << endl;
    cout << "[" << index << "] lower_text: " << lower_text << endl;
    cout << "[" << index << "] cleaned_upper: [" << cleaned_upper << "]" << endl;
    cout << "[" << index << "] cleaned_lower: [" << cleaned_lower << "]" << endl;

    //5. 정규식 검증
    regex upper_pattern("^[0-9]{2}$");

    bool upper_valid = regex_match(cleaned_upper, upper_pattern);
    bool lower_valid = validate_lower_pattern(cleaned_lower);

    cout << "[" << index << "] upper_valid: " << upper_valid << endl;
    cout << "[" << index << "] lower_valid: " << lower_valid << endl;

    // 6. 판단
    result.lpNum = cleaned_upper + cleaned_lower;

    if (lower_valid && upper_valid) { // 상,하단 모두 부합
        result.reliability = 1;
        cout << "정규식 일치" << endl;
    }
    else if (lower_valid) {          // 하단만 부합
        result.reliability = 0;
        cout << "정규식 일부 일치" << endl;
    }
    else                            // 규칙 위배
        result.reliability = -1;

    result.success = (result.reliability == 1);

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

    for (const auto& entry : fs::directory_iterator("../images")) {
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