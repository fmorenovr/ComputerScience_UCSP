#include "mainwindow.h"
#include "utils/bmp.h"
#include "utils/filters.h"
#include "utils/linearTransform.h"
#include "utils/geometricTransform.h"

void MainWindow::showImageM(unsigned char * d, int w, int h, QLabel * lbl, int formatT = 1){
    if(d != nullptr){
        int wdt = w;
        int hgh = h;
        QPixmap pix;
        if(formatT == 1){
            QImage image(w, h, QImage::Format_RGB888);
            for(int x = 0; x < image.width(); x++){
                for(int y = 0; y < image.height(); y++){
                    image.setPixelColor(x, y, QColor(d[y*wdt*3 + x*3 +0],d[y*wdt*3+x*3 +1],d[y*wdt*3+x*3 +2]));
                }
            }
            pix = QPixmap::fromImage(image);
        }
        else if (formatT == 0) {
            QImage image(w, h, QImage::Format_Grayscale8);
            for(int x = 0; x < image.width(); x++){
                for(int y = 0; y < image.height(); y++){
                     image.setPixelColor(x, y, QColor(d[y*wdt + x], d[y*wdt + x], d[y*wdt + x]));
                }
            }
            pix = QPixmap::fromImage(image);
        }
        //lbl->resize(w, h);
        lbl->setPixmap(pix);
    }
}
void MainWindow::showImage3Chann(unsigned char * d, int w, int h, std::vector<QLabel * > labels){
    if(d != nullptr){
        QImage im1(w, h, QImage::Format_Grayscale8);
        QImage im2(w, h, QImage::Format_Grayscale8);
        QImage im3(w, h, QImage::Format_Grayscale8);
        for(int x = 0; x < im1.width(); x++){
            for(int y = 0; y < im1.height(); y++){
                 im1.setPixelColor(x, y, QColor(d[y*w*3 + 3*x], d[y*w*3 + 3*x], d[y*w*3 + 3*x]));
                 im2.setPixelColor(x, y, QColor(d[y*w*3 + 3*x + 1], d[y*w*3 + 3*x + 1], d[y*w*3 + 3*x + 1]));
                 im3.setPixelColor(x, y, QColor(d[y*w*3 + 3*x + 2], d[y*w*3 + 3*x + 2], d[y*w*3 + 3*x + 2]));
            }
        }
        QPixmap pix1 = QPixmap::fromImage(im1);
        QPixmap pix2 = QPixmap::fromImage(im2);
        QPixmap pix3 = QPixmap::fromImage(im3);
        labels[0]->setPixmap(pix1);
        labels[1]->setPixmap(pix2);
        labels[2]->setPixmap(pix3);
    }
}
void MainWindow::on_pushButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("choose"), "", tr("Images (*.png *.jpg *.jpeg *.bmp *.gif)"));

    ImageFormatHeader head;
    if(QString::compare(filename, QString()) != 0){
        //int  height, width, size;
        int size;
        if(data != nullptr)
            delete [] data;
        data = ReadBMP(filename.toStdString(), height, width, size);

        head.set(height, width, size);
        //unsigned char * data = readBMP(filename.toStdString(), head);
        //RGB_ ** mat = getImageMatrix(filename.toStdString(), head);
        if(data != nullptr){
            int wdt = head.width;
            int hgh = head.heigh;

            QImage image(head.width, head.heigh, QImage::Format_RGB888);
            for(int x = 0; x < image.width(); x++){
                for(int y = 0; y < image.height(); y++){
                    //image.setPixelColor(x, y, QColor(0, 255, 0));
                    image.setPixelColor(x, y, QColor(data[y*wdt*3 + x*3 +0],data[y*wdt*3+x*3 +1],data[y*wdt*3+x*3 +2]));
                    //image.setPixelColor(x,y,QColor(mat[y][x].red, mat[y][x].green, mat[y][x].blue));
                }
            }
            //bool valid = image.load(filename);
            QPixmap pix = QPixmap::fromImage(image);
            //bool valid = pix->loadFromData(QByteArray::fromRawData((const char*)data, sizeImage));
            //bool valid = image.loadFromData(data, sizeImage);
            //if(valid){
                ui->labelImage->setPixmap(pix);
                //ui->
            //}
            //else{
            //    std::cout<<"Error loading image"<<std::endl;
            //}
            //delete data;
        }
    }

    //unsigned char * data = readBMP("../Other files/cat50x50.bmp",sizeImage);
    //delete data;
}

void MainWindow::on_polFilter_clicked()
{
    std::vector<double> coef;
    bool ok;
    double a = ui->p2Edit->toPlainText().toDouble(&ok);
    if(ok)
        coef.push_back(a);
    else
        coef.push_back(0);
    a = ui->p1Edit->toPlainText().toDouble(&ok);
    if(ok)
        coef.push_back(a);
    else
        coef.push_back(1);
    a = ui->p0Edit->toPlainText().toDouble(&ok);
    if(ok)
        coef.push_back(a);
    else
        coef.push_back(0);

    unsigned char * filtered = PolinomialTransform(coef, data, width, height, 3);
    showImageM(filtered, width, height, ui->labelPolFilter);
    delete filtered;
}

void MainWindow::on_pushButton_2_clicked()
{
    signed char kernellGauss[9] = {1, 3, 1, 3, 9, 3, 1, 3, 1};
    signed char kernellPerfil[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    int v = ui->horizontalSlider->value();
    //std::cout<<"Val->"<<v<<std::endl;
    if(v >= 0){
        unsigned char * convdata = nullptr;
        unsigned char * otherData = nullptr;
        for(int pp = 0; pp < v; pp++){
            if(pp == 0)
                convdata = convolutionGPU(data, width, height, kernellGauss, 3, 3);
            else{
                if(pp % 2 != 0){
                    otherData = convolutionGPU(convdata, width, height, kernellGauss, 3, 3);
                    delete [] convdata;
                }
                else{
                    convdata = convolutionGPU(otherData, width, height, kernellGauss, 3, 3);
                    delete [] otherData;
                }

            }

        }
        if(v == 0)
            showImageM(data, width, height, ui->labelConv);
        else if (v % 2 != 0){
            showImageM(convdata, width, height, ui->labelConv);
            delete [] convdata;
        }
        else{
            showImageM(otherData, width, height, ui->labelConv);
            delete [] otherData;
        }
    }
    else{
        unsigned char * convdata = nullptr;
        unsigned char * otherData = nullptr;
        v = -v;
        for(int pp = 0; pp < v; pp++){
            if(pp == 0)
                convdata = convolutionGPU(data, width, height, kernellPerfil, 3, 3);
            else{
                if(pp % 2 != 0){
                    otherData = convolutionGPU(convdata, width, height, kernellPerfil, 3, 3);
                    delete [] convdata;
                }
                else{
                    convdata = convolutionGPU(otherData, width, height, kernellPerfil, 3, 3);
                    delete [] otherData;
                }
            }

        }
        if (v % 2 != 0){
            showImageM(convdata, width, height, ui->labelConv);
            delete [] convdata;
        }
        else{
            showImageM(otherData, width, height, ui->labelConv);
            delete [] otherData;
        }
    }

    //showImageM(convdata, width, height, ui->labelConv);

    //delete [] convdata;
}

void MainWindow::on_pushButton_3_clicked()
{
    unsigned char * graydata = SimpleEdgeDetector(data, width, height, 100, 200);
    showImageM(graydata, width, height, ui->labelCanny, 0);
    delete  [] graydata;
}

void MainWindow::on_pushButton_4_clicked()
{
    unsigned char * grayscale = RGB2Gray(data, width*height*3);
    //unsigned char * real = DFTimageS(grayscale, width, height);
    unsigned char * real = DFTimageCuda(grayscale, width, height);
    std::vector<double> c{2.5, 0};
    unsigned char * highBrigh = PolinomialTransform(c, real, width, height, 1);
    showImageM(highBrigh, width, height, ui->labelFourier, 0);
    delete [] grayscale;
    delete [] real;
    delete [] highBrigh;
}

void MainWindow::on_pushButton_5_clicked()
{
    unsigned char * hsvData = RGB2HSV(data, height*width);
    std::vector<QLabel*> lab{ui->labelH, ui->labelS, ui->labelV};
    showImage3Chann(hsvData, width, height, lab);
    delete [] hsvData;
}

void MainWindow::on_pushButton_6_clicked(){
    int x1[4] = {0, width-1, 0, width-1};
    int y1[4] = {0, 0, height-1, height-1};
    int x2[4] = {2*width/8, 7*width/8, 0, width-1};
    int y2[4] = {0, 0, height-1, height-1};


    unsigned char * bilinear = new unsigned char[width*height*3];
    Bilinear(data, bilinear, x1, y1, x2, y2, width, height);
    //ui->labelBilinear->clear();
    showImageM(bilinear, width, height, ui->labelBilinear);
    std::cout<<"Previous to delete\n";
    delete [] bilinear;
}

// Convolution of Perfilate & Gauss
void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    //std::cout<<position<<"Pos"<<std::endl;
    if(data != nullptr)
        on_pushButton_2_clicked();
}
