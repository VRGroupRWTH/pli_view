/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_PLOT_SPECTROGRAM_H
#define QWT_PLOT_SPECTROGRAM_H

#include "third_party/qwt/qwt_global.h"
#include "third_party/qwt/qwt_raster_data.h"
#include "third_party/qwt/qwt_plot_rasteritem.h"
#include <qlist.h>

class QwtColorMap;

/*!
  \brief A plot item, which displays a spectrogram

  A spectrogram displays 3-dimensional data, where the 3rd dimension
  ( the intensity ) is displayed using colors. The colors are calculated
  from the values using a color map.

  On multi-core systems the performance of the image composition
  can often be improved by dividing the area into tiles - each of them
  rendered in a different thread ( see QwtPlotItem::setRenderThreadCount() ).

  In ContourMode contour lines are painted for the contour levels.

  \image html spectrogram3.png

  \sa QwtRasterData, QwtColorMap, QwtPlotItem::setRenderThreadCount()
*/

class QWT_EXPORT QwtPlotSpectrogram: public QwtPlotRasterItem
{
public:
    /*!
      The display mode controls how the raster data will be represented.
      \sa setDisplayMode(), testDisplayMode()
    */

    enum DisplayMode
    {
        //! The values are mapped to colors using a color map.
        ImageMode = 0x01,

        //! The data is displayed using contour lines
        ContourMode = 0x02
    };

    //! Display modes
    typedef QFlags<DisplayMode> DisplayModes;

    explicit QwtPlotSpectrogram( const QString &title = QString::null );
    virtual ~QwtPlotSpectrogram();

    void setDisplayMode( DisplayMode, bool on = true );
    bool testDisplayMode( DisplayMode ) const;

    void setData( QwtRasterData *data );
    const QwtRasterData *data() const;
    QwtRasterData *data();

    void setColorMap( QwtColorMap * );
    const QwtColorMap *colorMap() const;

    virtual QwtInterval interval(Qt::Axis) const;
    virtual QRectF pixelHint( const QRectF & ) const;

    void setDefaultContourPen( const QColor &, 
        qreal width = 0.0, Qt::PenStyle = Qt::SolidLine );
    void setDefaultContourPen( const QPen & );
    QPen defaultContourPen() const;

    virtual QPen contourPen( double level ) const;

    void setConrecFlag( QwtRasterData::ConrecFlag, bool on );
    bool testConrecFlag( QwtRasterData::ConrecFlag ) const;

    void setContourLevels( const QList<double> & );
    QList<double> contourLevels() const;

    virtual int rtti() const;

    virtual void draw( QPainter *p,
        const QwtScaleMap &xMap, const QwtScaleMap &yMap,
        const QRectF &rect ) const;

protected:
    virtual QImage renderImage(
        const QwtScaleMap &xMap, const QwtScaleMap &yMap,
        const QRectF &area, const QSize &imageSize ) const;

    virtual QSize contourRasterSize(
        const QRectF &, const QRect & ) const;

    virtual QwtRasterData::ContourLines renderContourLines(
        const QRectF &rect, const QSize &raster ) const;

    virtual void drawContourLines( QPainter *p,
        const QwtScaleMap &xMap, const QwtScaleMap &yMap,
        const QwtRasterData::ContourLines& lines ) const;

    void renderTile( const QwtScaleMap &xMap, const QwtScaleMap &yMap,
        const QRect &imageRect, QImage *image ) const;

private:
    class PrivateData;
    PrivateData *d_data;
};

Q_DECLARE_OPERATORS_FOR_FLAGS( QwtPlotSpectrogram::DisplayModes )

#endif
