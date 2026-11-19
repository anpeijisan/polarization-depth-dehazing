import sys
import json
import time
import threading
import traceback
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pyqtgraph as pg
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt, QObject, Signal, QTimer, QRectF, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QLinearGradient, QPen, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
 QGridLayout,QFrame,QLabel,QPushButton,QComboBox,QDoubleSpinBox,QMessageBox,QSizePolicy)
from polar_camera import PolarCamera
from paper_metrics import METRICS
ROOT=Path(__file__).resolve().parent
METRIC_COLORS=dict(zip(METRICS,('#C6AACD','#F08474','#C6AACD','#F08474','#FFDF96','#BED89E')))
TEXT_COLOR='#142f43'
FONT_FAMILY='Times New Roman'
ANGLE_FONT=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',14)
PANEL_OVERHEAD=32
CHANGE_COLORS=np.array([[59,76,192],[247,247,247],[180,4,38]],dtype=np.float32)

def depth_change_image(before,after,span,limit):
    change=np.clip((after-before)/(span*limit),-1,1)[...,None]
    colors=np.where(change<0,CHANGE_COLORS[1]+change*(CHANGE_COLORS[1]-CHANGE_COLORS[0]),
                    CHANGE_COLORS[1]+change*(CHANGE_COLORS[2]-CHANGE_COLORS[1]))
    return np.uint8(np.rint(colors))

def make_angle_montage(angles,tile_width=160):
    """Show the four analyzer images in a 2-by-2 grid at a shared scale."""
    values=np.uint8(np.clip(angles*2,0,1)*255)
    values=cv2.resize(values,(tile_width,max(1,round(values.shape[0]*tile_width/values.shape[1]))),interpolation=cv2.INTER_AREA)
    labels=('0°','45°','90°','135°')
    tiles=[]
    for i,label in enumerate(labels):
        tile=Image.fromarray(values[:,:,i]);draw=ImageDraw.Draw(tile)
        text_box=draw.textbbox((0,0),label,font=ANGLE_FONT)
        draw.rectangle((2,2,text_box[2]+10,text_box[3]+5),fill=245)
        draw.text((6,2),label,font=ANGLE_FONT,fill=20)
        tiles.append(np.asarray(tile))
    image=np.concatenate((np.concatenate((tiles[0],tiles[1]),axis=1),
                          np.concatenate((tiles[2],tiles[3]),axis=1)),axis=0)
    image[image.shape[0]//2-1:image.shape[0]//2+1,:]=235
    image[:,image.shape[1]//2-1:image.shape[1]//2+1]=235
    return image

class Signals(QObject):
    result=Signal(object); status=Signal(str); error=Signal(str); finished=Signal()

class ImageView(QLabel):
    def __init__(self,empty='Waiting for data'):
        super().__init__(empty);self.setObjectName('image');self.setAlignment(Qt.AlignCenter);self.setMinimumSize(70,55);self.pix=None
        self.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
    def reset(self,text='Waiting for data'):
        self.pix=None;self.clear();self.setText(text)
    def show_image(self,image):
        image=np.ascontiguousarray(image);h,w=image.shape[:2]
        fmt=QImage.Format_Grayscale8 if image.ndim==2 else QImage.Format_RGB888
        self.pix=QPixmap.fromImage(QImage(image.data,w,h,image.strides[0],fmt).copy());self.update_pix()
    def update_pix(self):
        if self.pix is not None:self.setPixmap(self.pix.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def resizeEvent(self,event):
        super().resizeEvent(event);self.update_pix()
    def sizeHint(self):
        return QSize(160,160)

class ColorBar(QWidget):
    def __init__(self,kind='depth'):
        super().__init__();self.kind=kind;self.lo=None;self.hi=None;self.setFixedWidth(22 if kind=='signed' else 16);self.setMinimumHeight(80)
        self.setFont(QFont(FONT_FAMILY,9,QFont.Bold));self.setStyleSheet('font-family:"Times New Roman";font-size:12px;font-weight:700')
    def set_range(self,lo,hi):
        self.lo=lo;self.hi=hi
        self.top_text=f'{hi:g}'+(' Far' if self.kind=='depth' else '')
        self.bottom_text=f'{lo:g}'+(' Near' if self.kind=='depth' else '')
        self.setFixedWidth(10+max(self.fontMetrics().horizontalAdvance(text) for text in (self.top_text,self.bottom_text)))
        self.update()
    def reset(self):self.lo=None;self.hi=None;self.update()
    def paintEvent(self,event):
        if self.lo is None:return
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);top=0;bottom=max(1,self.height()-3);x=0;w=6
        grad=QLinearGradient(0,bottom,0,top)
        if self.kind=='signed':
            for stop,color in zip([0,.5,1],CHANGE_COLORS):grad.setColorAt(stop,QColor(*map(int,color)))
        else:
            for i in range(11):
                v=int(i*255/10);c=cv2.applyColorMap(np.array([[v]],dtype=np.uint8),cv2.COLORMAP_INFERNO)[0,0]
                grad.setColorAt(i/10,QColor(int(c[2]),int(c[1]),int(c[0])))
        p.setPen(Qt.NoPen);p.setBrush(grad);p.drawRoundedRect(QRectF(x,top,w,bottom-top),3,3)
        p.setPen(QPen(QColor(TEXT_COLOR)));p.drawText(8,p.fontMetrics().ascent(),self.top_text);p.drawText(8,bottom,self.bottom_text)
        if self.kind=='signed':p.drawText(8,int((top+bottom)/2)+4,'0')

class Card(QFrame):
    def __init__(self,title):
        super().__init__();self.setObjectName('card');self.layout=QVBoxLayout(self);self.layout.setContentsMargins(4,3,4,3);self.layout.setSpacing(2)
        header=QWidget();header.setFixedHeight(24);row=QHBoxLayout(header);row.setContentsMargins(0,0,0,0)
        self.title=QLabel(title);self.title.setObjectName('cardTitle');row.addWidget(self.title);row.addStretch()
        self.header=row;self.layout.addWidget(header)
        body_host=QWidget();body_host.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.body=QVBoxLayout(body_host);self.body.setContentsMargins(0,0,0,0);self.body.setSpacing(3);self.layout.addWidget(body_host,1)

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('Polarimetric Dehazing & Depth Estimation');self.resize(980,830);self.setMinimumSize(980,780)
        for font in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
        QApplication.instance().setFont(QFont(FONT_FAMILY,10,QFont.Bold))
        self.setFont(QFont(FONT_FAMILY,10,QFont.Bold))
        self.signals=Signals();self.signals.result.connect(self.on_result);self.signals.status.connect(self.status);self.signals.error.connect(self.on_error);self.signals.finished.connect(self.on_finished)
        self.stop_flag=threading.Event();self.worker=None;self.current=None;self.depth_range=None;self.image_ratio=1.0
        self.display_times=deque(maxlen=31);self.display_fps=None
        root=QWidget();root.setObjectName('root');self.setCentralWidget(root);outer=QVBoxLayout(root);outer.setContentsMargins(8,6,8,6);outer.setSpacing(4)
        self.state=QLabel('Ready');self.state.setObjectName('state');self.state.setWordWrap(True)
        self.grid_host=QWidget();self.grid_host.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored);outer.addWidget(self.grid_host,1)
        content=QHBoxLayout(self.grid_host);content.setContentsMargins(0,0,0,0);content.setSpacing(4)
        content.setAlignment(Qt.AlignHCenter|Qt.AlignTop);self.content_layout=content
        self.sidebar_host=QWidget();sidebar=QVBoxLayout(self.sidebar_host);sidebar.setContentsMargins(0,0,0,0);sidebar.setSpacing(4)
        self.results_host=QWidget();grid=QGridLayout(self.results_host);grid.setContentsMargins(0,0,0,0);grid.setSpacing(4)
        grid.setAlignment(Qt.AlignLeft|Qt.AlignTop);self.content_grid=grid
        content.addWidget(self.sidebar_host,0,Qt.AlignTop);content.addWidget(self.results_host,0,Qt.AlignTop)
        self.angle_card=Card('Polarization Inputs');self.metric_card=Card('Metrics After Dehazing')
        self.angle_card.layout.setContentsMargins(0,3,0,3)
        self.raw_card=Card('Hazy Intensity S<sub>0</sub>');self.depth0_card=Card('Depth Before Dehazing')
        self.enh_card=Card('Dehazed Intensity S<sub>0</sub>');self.depth1_card=Card('Depth After Dehazing')
        self.delta_card=Card('Depth Difference (After − Before)')
        self.delta_card.setToolTip('Signed depth difference: (after - before) / shared depth range. Blue: decrease. Red: increase.')
        sidebar.addWidget(self.angle_card)
        self.control_panel=Card('Controls & Performance');self.control_panel.setObjectName('controlPanel')
        self.control_panel.layout.setContentsMargins(12,12,12,12);self.control_panel.layout.setSpacing(10)
        controls=self.control_panel.body;controls.setSpacing(10)
        source_field=QVBoxLayout();source_field.setSpacing(5);source_field.addWidget(QLabel('Source'))
        self.source=QComboBox();self.source.addItems(['Sample · Synthetic fog','Camera · Live','Road · First half']);self.source.currentIndexChanged.connect(self.source_changed);source_field.addWidget(self.source);controls.addLayout(source_field)
        exposure_field=QVBoxLayout();exposure_field.setSpacing(5);exposure_field.addWidget(QLabel('Exposure / μs'))
        self.exposure=QDoubleSpinBox();self.exposure.setRange(1,10000000);self.exposure.setDecimals(0);self.exposure.setValue(2000);exposure_field.addWidget(self.exposure);controls.addLayout(exposure_field)
        buttons=QHBoxLayout();buttons.setSpacing(12)
        self.start_btn=QPushButton('Start');self.start_btn.setObjectName('primary');self.start_btn.clicked.connect(self.start);buttons.addWidget(self.start_btn,1)
        self.stop_btn=QPushButton('Stop');self.stop_btn.setEnabled(False);self.stop_btn.clicked.connect(self.stop);buttons.addWidget(self.stop_btn,1);controls.addLayout(buttons)
        self.scale_btn=QPushButton('Reset Depth Scale');self.scale_btn.clicked.connect(self.reset_scale);controls.addWidget(self.scale_btn)
        range_field=QVBoxLayout();range_field.setSpacing(5);range_field.addWidget(QLabel('Depth Difference Range'))
        self.diff_scale=QComboBox();self.diff_scale.addItems(['±0.10','±0.25','±0.50','±1.00']);self.diff_scale.setCurrentIndex(3);self.diff_scale.currentIndexChanged.connect(self.refresh_diff_scale);range_field.addWidget(self.diff_scale);controls.addLayout(range_field)
        self.timing_panel=QFrame();self.timing_panel.setObjectName('timingPanel')
        timings=QGridLayout(self.timing_panel);timings.setContentsMargins(8,6,8,6);timings.setHorizontalSpacing(8);timings.setVerticalSpacing(2)
        timing_title=QLabel('Processing Time');timing_title.setObjectName('timingTitle');timings.addWidget(timing_title,0,0)
        unit=QLabel('ms');unit.setAlignment(Qt.AlignRight|Qt.AlignVCenter);timings.addWidget(unit,0,1)
        self.timing_values={}
        for row,(key,name) in enumerate((('dehazing','Dehazing'),('depth','Depth Estimation'),('metrics','Metrics'),('rendering','Rendering')),1):
            timings.addWidget(QLabel(name),row,0)
            value=QLabel('—');value.setAlignment(Qt.AlignRight|Qt.AlignVCenter);value.setMinimumWidth(45)
            timings.addWidget(value,row,1);timings.setRowStretch(row,1);self.timing_values[key]=value
        self.timing_values['depth'].setToolTip('Combined depth estimation for the hazy and dehazed images, including CPU readback.')
        self.timing_values['metrics'].setToolTip('Paired-reference metric computation; unavailable for live camera without a reference.')
        self.timing_values['rendering'].setToolTip('Image conversion, depth color mapping, metric chart updates and completed GUI painting.')
        controls.addWidget(self.timing_panel,1)
        self.performance_panel=QFrame();self.performance_panel.setObjectName('performancePanel')
        performance=QVBoxLayout(self.performance_panel);performance.setContentsMargins(10,8,10,8);performance.setSpacing(6)
        fps_row=QHBoxLayout();fps_row.addWidget(QLabel('Display FPS'));fps_row.addStretch()
        self.fps_label=QLabel('—');self.fps_label.setObjectName('fpsValue');fps_row.addWidget(self.fps_label);performance.addLayout(fps_row)
        self.fps_label.setToolTip('Full-pipeline display rate, measured after frame acquisition, restoration, depth estimation, metrics and GUI painting. Rolling window: up to 30 displayed frames.')
        performance.addWidget(self.state);controls.addWidget(self.performance_panel)
        sidebar.addWidget(self.control_panel)
        for card,row,col in [(self.raw_card,0,0),(self.enh_card,0,1),(self.depth0_card,1,0),(self.depth1_card,1,1),
                             (self.metric_card,2,0),(self.delta_card,2,1)]:
            grid.addWidget(card,row,col,Qt.AlignTop)
        self.angle_view=ImageView();self.angle_card.body.addWidget(self.angle_view,1,Qt.AlignLeft)
        self.raw=ImageView();self.enh=ImageView();self.dep0=ImageView();self.dep1=ImageView();self.delta=ImageView()
        self.bar0=ColorBar();self.bar1=ColorBar();self.delta_bar=ColorBar('signed')
        self.views=[self.raw,self.dep0,self.enh,self.dep1,self.delta]
        self.image_cards=[self.raw_card,self.depth0_card,self.enh_card,self.depth1_card,self.delta_card]
        self.image_scales=[None,self.bar0,None,self.bar1,self.delta_bar]
        for card,view,cbar in zip(self.image_cards,self.views,self.image_scales):
            r=QHBoxLayout();r.setContentsMargins(0,0,0,0);r.setSpacing(3)
            r.setAlignment(Qt.AlignLeft)
            r.addWidget(view)
            if cbar is not None:r.addWidget(cbar)
            card.body.addLayout(r,1)
        self.delta_note=QLabel('Red (+) indicates increased predicted depth (farther)\nBlue (−) indicates decreased predicted depth (nearer)')
        self.delta_note.setObjectName('deltaCaption');self.delta_note.setTextFormat(Qt.PlainText)
        self.delta_note.setWordWrap(True);self.delta_note.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.delta_card.body.addWidget(self.delta_note)
        self.chart=pg.PlotWidget();self.chart.setBackground(None);self.chart.setStyleSheet('background:transparent');self.chart.showGrid(x=False,y=True,alpha=.15)
        self.chart.setLabel('left','After dehazing',**{'font-family':FONT_FAMILY,'font-weight':'bold','font-size':'13px','color':TEXT_COLOR})
        for axis in ['left','bottom']:self.chart.getAxis(axis).setTextPen(TEXT_COLOR);self.chart.getAxis(axis).setPen(pg.mkPen(TEXT_COLOR))
        self.chart.setMouseEnabled(x=False,y=False);self.chart.hideButtons();self.chart.setMenuEnabled(False)
        self.chart.setLimits(xMin=-.5,xMax=5.5);self.chart.setXRange(-.5,5.5,padding=0)
        for axis in ['left','bottom']:self.chart.getAxis(axis).setTickFont(QFont(FONT_FAMILY,9,QFont.Bold))
        self.chart.getAxis('left').setWidth(38)
        self.chart.getAxis('left').label.setFont(QFont(FONT_FAMILY,10,QFont.Bold))
        self.metric_order=[2,3,4,5,1,0]
        self.update_metric_ticks()
        self.metric_card.body.addWidget(self.chart,1,Qt.AlignLeft)
        self.chart.setToolTip('Raw values after dehazing on one linear axis. Fixed order: NMAE, AbsRel, SqRel, RMSE, SSIM, PSNR. Errors ↓; SSIM ↑ and PSNR ↑.')
        self.metric_status=QLabel('No reference',self.chart);self.metric_status.setObjectName('state');self.metric_status.setAlignment(Qt.AlignCenter)
        self.update_metrics(None)
        self.setStyleSheet('''
            QWidget{font-family:"Times New Roman";font-size:14px;font-weight:700;color:#142f43}
            #root,QMainWindow{background:#e9eff2}
            QLabel{background:transparent}
            #card{background:transparent;border:none;border-radius:0}
            #controlPanel{background:#fff;border:1px solid #d8e3e8;border-radius:8px}
            #controlPanel QWidget{font-size:12px}
            #controlPanel #cardTitle{font-size:14px}
            #controlPanel #timingTitle{font-size:14px}
            #timingPanel{background:#f6f9fb;border:1px solid #c4d5df;border-radius:6px}
            #performancePanel{background:#edf4f7;border:1px solid #c4d5df;border-radius:6px}
            #cardTitle{font-size:16px;font-weight:700;color:#142f43}
            #deltaCaption{font-size:12px;font-weight:700;color:#142f43}
            #image{background:transparent;color:#142f43;border-radius:0}
            #state{font-size:14px;font-weight:700;color:#142f43}
            #controlPanel #fpsValue{font-size:20px;font-weight:700;color:#102b40}
            #controlPanel #state{font-size:12px;font-weight:700;color:#142f43}
            QComboBox,QDoubleSpinBox{background:#fff;color:#142f43;border:1px solid #b7cbd3;border-radius:5px;padding:4px 8px;min-height:22px}
            QComboBox:disabled,QDoubleSpinBox:disabled{color:#142f43;background:#f0f4f6}
            QComboBox QAbstractItemView{color:#142f43;background:#fff;selection-background-color:#dcefe9;selection-color:#142f43}
            QPushButton{background:#fff;border:1px solid #b7cbd3;border-radius:5px;padding:8px 12px;color:#142f43}
            QPushButton:hover{background:#e1f2ed}
            QPushButton:disabled{color:#142f43;background:#edf2f4;border-style:dashed}
            QPushButton#primary{background:#b9e3d7;border:1px solid #6baf9c;color:#102b40;font-weight:700}
            QToolTip{font-size:14px;font-weight:700;color:#142f43;background:#fff;border:1px solid #b7cbd3}
        ''')
        self.source.setCurrentIndex(2);self.state.setText('Ready')
    def showEvent(self,event):
        super().showEvent(event);QTimer.singleShot(0,self.sync_image_sizes)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'views'):QTimer.singleShot(0,self.sync_image_sizes)
    def sync_image_sizes(self):
        # The complete four-angle montage, chart, and result images share one size.
        gap=self.content_grid.spacing();scale_width=max(bar.width() for bar in (self.bar0,self.bar1,self.delta_bar))+3;ratio=self.image_ratio
        available_width=self.grid_host.width();available_height=self.grid_host.height()
        caption_height=2*self.delta_note.fontMetrics().lineSpacing()+2
        caption_extra=caption_height+self.delta_card.body.spacing()
        image_width=max(70,min((available_width-2*(8+scale_width)-2*gap)//3,
            int((available_height-3*PANEL_OVERHEAD-2*gap-caption_extra)*ratio/3)))
        card_width=image_width+8+scale_width
        height=max(70,int(image_width/ratio))
        for card,view,bar in zip(self.image_cards,self.views,self.image_scales):
            view.setFixedSize(image_width,height)
            if bar is not None:bar.setFixedHeight(height)
            card.setFixedSize(card_width,height+PANEL_OVERHEAD+(caption_extra if card is self.delta_card else 0))
        self.delta_note.setFixedSize(card_width-8,caption_height)
        # The bottom axis occupies its own strip below the aligned image baseline.
        chart_extra=int(self.chart.getAxis('bottom').height())
        self.metric_card.setFixedSize(card_width,height+PANEL_OVERHEAD+chart_extra)
        self.chart.setFixedSize(card_width-8,height+chart_extra)
        self.metric_status.setGeometry(self.chart.rect())
        self.update_metric_ticks()
        total_height=3*(height+PANEL_OVERHEAD)+2*gap+caption_extra
        self.results_host.setFixedSize(2*card_width+gap,total_height)
        sidebar_width=image_width
        angle_width=image_width;angle_height=height
        self.angle_view.setFixedSize(angle_width,angle_height)
        self.angle_card.setFixedSize(sidebar_width,angle_height+PANEL_OVERHEAD)
        self.control_panel.setFixedSize(sidebar_width,total_height-self.angle_card.height()-gap)
        self.sidebar_host.setFixedSize(sidebar_width,total_height)
    def source_changed(self,index):
        if hasattr(self,'raw'):
            self.current=None;self.depth_range=None;self.update_metrics(None);self.reset_display_fps();self.update_processing_times(None)
            for view in [*self.views,self.angle_view]:view.reset()
            for bar in [self.bar0,self.bar1,self.delta_bar]:bar.reset()
            self.state.setText('Source changed · Ready')
    def status(self,text):self.state.setText(text)
    def reset_display_fps(self):
        self.display_times.clear();self.display_fps=None;self.fps_label.setText('—')
    def update_processing_times(self,timings):
        for key,label in self.timing_values.items():
            value=timings[key] if timings is not None else None
            label.setText('—' if value is None else f'{value:.1f}')
    def start(self):
        if self.worker and self.worker.is_alive():return
        self.current=None;self.depth_range=None;self.update_metrics(None);self.reset_display_fps();self.update_processing_times(None)
        for im in [*self.views,self.angle_view]:im.reset()
        for bar in [self.bar0,self.bar1,self.delta_bar]:bar.reset()
        self.stop_flag.clear();self.start_btn.setEnabled(False);self.stop_btn.setEnabled(True);self.source.setEnabled(False);self.exposure.setEnabled(False)
        self.worker=threading.Thread(target=self.run_worker,args=(self.source.currentIndex(),self.exposure.value()),daemon=True);self.worker.start()
    def reset_scale(self):
        self.depth_range=None
        if self.current is not None:self.render_result(self.current)
    def refresh_diff_scale(self):
        if self.current is not None:self.render_result(self.current)
    def stop(self):self.stop_flag.set();self.stop_btn.setEnabled(False);self.status('Stopping…')
    def run_worker(self,source,exposure):
        camera=None
        try:
            self.signals.status.emit('Loading models…');from pipeline import Pipeline
            pipeline=Pipeline()
            if self.stop_flag.is_set():return
            if source==0:
                data=np.load(ROOT/'sample/hazy_angles.npy',mmap_mode='r');refs=np.load(ROOT/'sample/paired_source.npz');ref_images=refs['image'];ref_depths=refs['depth'];idx=0
                if len(data)!=len(ref_images) or len(data)!=len(ref_depths):raise ValueError('Video and paired reference frame counts do not match.')
                self.signals.status.emit('Sample playback · Synthetic fog')
            elif source==2:
                bundle=ROOT/'sample/road_fog_square'
                data=np.load(bundle/'hazy_angles.npy',mmap_mode='r')
                refs=np.load(bundle/'paired_source.npz');ref_images=refs['image'];ref_depths=refs['depth']
                road_info=json.loads((bundle/'metadata.json').read_text(encoding='utf-8'))
                if len(data)!=len(ref_images) or len(data)!=len(ref_depths):raise ValueError('Road video and paired reference frame counts do not match.')
                road_count=road_info.get('playback_frame_count',len(data));idx=0
                self.signals.status.emit('Road first half · Sequential playback')
            else:
                camera=PolarCamera();camera.start(mode='continuous',exposure_us=exposure);self.signals.status.emit('Connecting to camera…')
            last=None
            while not self.stop_flag.is_set():
                frame_started_at=time.perf_counter()
                if camera:
                    frame=camera.get_latest()
                    if frame is None or frame is last:
                        if camera.error:raise RuntimeError(camera.error)
                        time.sleep(.02);continue
                    last=frame;angles=frame.angles.astype(np.float32)/510.0;frame_id=frame.frame_id;reference=None
                elif source==2:
                    i=idx%road_count;idx+=1
                    angles=np.moveaxis(data[i],0,-1)
                    frame_id=road_info['start_frame']+i;reference=(ref_images[i],ref_depths[i])
                else:
                    i=idx%len(data);angles=np.moveaxis(np.asarray(data[i]),0,-1);frame_id=idx;reference=(ref_images[i],ref_depths[i]);idx+=1
                display_shape=road_info['display_shape'] if source==2 else None
                out=pipeline.run(angles,reference,display_shape=display_shape,
                    preserve_aspect=source==2 and road_info.get('preserve_aspect',False))
                out['id']=frame_id;out['source']=source;out['frame_started_at']=frame_started_at;self.signals.result.emit(out)
                if not camera:self.stop_flag.wait(.02)
        except Exception:self.signals.error.emit(traceback.format_exc())
        finally:
            try:
                if camera:camera.stop()
            except Exception:self.signals.error.emit(traceback.format_exc())
            self.signals.finished.emit()
    def on_result(self,out):
        if self.stop_flag.is_set():return
        rendering_started_at=time.perf_counter()
        self.current=out;self.update_metrics(out['metrics'])
        height,width=out['source_shape'];ratio=width/height
        if abs(ratio-self.image_ratio)>1e-6:
            self.image_ratio=ratio;self.sync_image_sizes()
        self.render_result(out)
        # Complete the image and chart paints before counting a displayed frame.
        self.angle_view.repaint();self.results_host.repaint()
        displayed_at=time.perf_counter()
        out['display_timings_ms']=dict(out['timings_ms'],rendering=(displayed_at-rendering_started_at)*1000)
        self.update_processing_times(out['display_timings_ms'])
        if not self.display_times:self.display_times.append(out['frame_started_at'])
        self.display_times.append(displayed_at)
        self.display_fps=(len(self.display_times)-1)/(displayed_at-self.display_times[0])
        self.fps_label.setText(f'{self.display_fps:.2f}')
        self.state.setText(f"Frame {out['id']} · {(displayed_at-out['frame_started_at'])*1000:.0f} ms/frame")
    def render_result(self,out):
        scale_width=max(bar.width() for bar in (self.bar0,self.bar1,self.delta_bar))
        gray=lambda x:np.uint8(np.clip(x,0,1)*255)
        self.angle_view.show_image(make_angle_montage(out['angles'],max(1,self.angle_view.width()//2)))
        height,width=out['source_shape'];display_size=(max(1,round(width*256/height)),256)
        match_frame=lambda x:cv2.resize(x,display_size,interpolation=cv2.INTER_LINEAR)
        self.raw.show_image(match_frame(gray(out['raw'])));self.enh.show_image(match_frame(gray(out['enhanced'])))
        if self.depth_range is None:
            both=np.concatenate((out['before'].ravel(),out['after'].ravel()))
            lo,hi=np.percentile(both,[1,99]);self.depth_range=(float(lo),max(float(hi),float(lo)+1e-6))
        lo,hi=self.depth_range
        before=np.clip((out['before']-lo)/(hi-lo),0,1);after=np.clip((out['after']-lo)/(hi-lo),0,1)
        def heat(x):return cv2.cvtColor(cv2.applyColorMap(gray(x),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
        self.dep0.show_image(match_frame(heat(before)));self.dep1.show_image(match_frame(heat(after)));self.bar0.set_range(0,1);self.bar1.set_range(0,1)
        # Signed change uses the un-clipped predictions: after minus before.
        scale=(0.1,0.25,0.5,1.0)[self.diff_scale.currentIndex()]
        self.delta.show_image(match_frame(depth_change_image(out['before'],out['after'],hi-lo,scale)))
        self.delta_bar.set_range(-scale,scale)
        if max(bar.width() for bar in (self.bar0,self.bar1,self.delta_bar))!=scale_width:self.sync_image_sizes()
    def update_metric_ticks(self):
        font=QFont(FONT_FAMILY,8,QFont.Bold);font.setStretch(80)
        self.chart.getAxis('bottom').setTickFont(font)
        self.chart.getAxis('bottom').setHeight(24)
        self.chart.getAxis('bottom').setStyle(hideOverlappingLabels=False,textFillLimits=[(0,1.0)])
        labels=[METRICS[index]+('\u2009↑' if index<2 else '\u2009↓') for index in self.metric_order]
        self.chart.getAxis('bottom').setTicks([list(enumerate(labels))])
    def update_metrics(self,metrics):
        self.chart.clear()
        self.chart.setYRange(0,1,padding=0)
        if metrics is None:
            self.metric_order=[2,3,4,5,1,0];self.update_metric_ticks()
            self.metric_status.setText('No reference');self.metric_status.show();self.metric_status.raise_();return
        self.metric_status.hide()
        after=np.asarray(metrics[1],dtype=float)
        values=after[self.metric_order];valid=np.isfinite(values)
        if np.any(valid):
            low=min(0,float(np.min(values[valid])));high=max(0,float(np.max(values[valid])))
            span=high-low or 1.0
            self.chart.setYRange(low-(.1*span if low<0 else 0),high+.18*span,padding=0)
        for i,index in enumerate(self.metric_order):
            if not valid[i]:continue
            name=METRICS[index];value=after[index]
            self.chart.addItem(pg.BarGraphItem(x=[i],height=[value],width=.62,brush=pg.mkBrush(METRIC_COLORS[name]),pen=pg.mkPen('#888888',width=.8)))
            text=f'{value:.2f}' if name=='PSNR' else f'{value:.3f}'
            label=pg.TextItem(text,color=TEXT_COLOR,anchor=(.5,1 if value>=0 else 0));label.setFont(QFont(FONT_FAMILY,8,QFont.Bold))
            label.setPos(i,value);self.chart.addItem(label)
    def on_error(self,text):self.status('Processing failed');QMessageBox.critical(self,'Processing failed',text[-2500:])
    def on_finished(self):
        self.start_btn.setEnabled(True);self.stop_btn.setEnabled(False);self.source.setEnabled(True);self.exposure.setEnabled(True)
        if self.stop_flag.is_set():self.status('Stopped')
    def closeEvent(self,event):
        if self.worker and self.worker.is_alive():self.stop();event.ignore();QTimer.singleShot(300,self.try_close)
        else:event.accept()
    def try_close(self):
        if self.worker and self.worker.is_alive():QTimer.singleShot(300,self.try_close)
        else:self.close()
if __name__=='__main__':
    app=QApplication(sys.argv);window=Window();window.show();sys.exit(app.exec())


