#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QLabel>

extern "C"
unsigned char * DFTimageCuda(unsigned char * data, int width, int height);

extern "C"
unsigned char * RGB2HSV(unsigned char * data, int npixels);

extern "C"
unsigned char * convolutionGPU(unsigned char * rowdata, int width, int height, signed char * kernell, int kernelSize, int nchann);

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_polFilter_clicked();
    void showImageM(unsigned char * data, int w, int h, QLabel * l, int type);
    void showImage3Chann(unsigned char * data, int w, int h, std::vector<QLabel*> labels);

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_horizontalSlider_sliderMoved(int position);

private:
    unsigned char * data = nullptr;
    int width, height;
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
