#ifndef PLI_VIS_SPAN_SLIDER_HPP
#define PLI_VIS_SPAN_SLIDER_HPP

#include <QObject>
#include <QSlider>
#include <QStyle>

#define QXT_INIT_PRIVATE(PUB) qxt_d.setPublic(this);

class QStylePainter;
class QStyleOptionSlider;

template <typename public_class>
class QxtPrivate
{
public:
  virtual ~QxtPrivate() {}

  void QXT_setPublic(public_class* pub) { qxt_p_ptr = pub; }

protected:
  public_class& qxt_p() { return *qxt_p_ptr; }
  const public_class& qxt_p() const { return *qxt_p_ptr; }
  public_class* qxt_ptr() { return qxt_p_ptr; }
  const public_class* qxt_ptr() const { return qxt_p_ptr; }

private:
  public_class* qxt_p_ptr;
};
template <typename public_class, typename private_class>
class private_interface
{
  friend class QxtPrivate<public_class>;

public:
  private_interface()
  {
    private_ = new private_class;
  }
  private_interface(const private_interface&) = delete;
  ~private_interface()
  {
    delete private_;
  }

  private_interface& operator=(const private_interface&) = delete;

  void setPublic(public_class* pub)
  {
    private_->QXT_setPublic(pub);
  }

  private_class& operator()() { return *static_cast<private_class*>(private_); }
  const private_class& operator()() const { return *static_cast<private_class*>(private_); }
  private_class* operator->() { return  static_cast<private_class*>(private_); }
  const private_class* operator->() const { return  static_cast<private_class*>(private_); }

private:
  QxtPrivate<public_class>* private_ = nullptr;
};

class QxtSpanSliderPrivate;

class QxtSpanSlider : public QSlider
{
  Q_OBJECT
  Q_PROPERTY(int lowerValue READ lowerValue WRITE setLowerValue)
  Q_PROPERTY(int upperValue READ upperValue WRITE setUpperValue)
  Q_PROPERTY(int lowerPosition READ lowerPosition WRITE setLowerPosition)
  Q_PROPERTY(int upperPosition READ upperPosition WRITE setUpperPosition)
  Q_PROPERTY(HandleMovementMode handleMovementMode READ handleMovementMode WRITE setHandleMovementMode)
  Q_ENUMS   (HandleMovementMode)

public:
  explicit QxtSpanSlider(QWidget* parent = 0);
  explicit QxtSpanSlider(Qt::Orientation orientation, QWidget* parent = 0);
  virtual ~QxtSpanSlider();

  enum HandleMovementMode
  {
    FreeMovement,
    NoCrossing,
    NoOverlapping
  };

  enum SpanHandle
  {
    NoHandle,
    LowerHandle,
    UpperHandle
  };

  HandleMovementMode handleMovementMode() const;
  void setHandleMovementMode(HandleMovementMode mode);

  int lowerValue() const;
  int upperValue() const;

  int lowerPosition() const;
  int upperPosition() const;

  public Q_SLOTS:
  void setLowerValue(int lower);
  void setUpperValue(int upper);
  void setSpan(int lower, int upper);

  void setLowerPosition(int lower);
  void setUpperPosition(int upper);

Q_SIGNALS:
  void spanChanged(int lower, int upper);
  void lowerValueChanged(int lower);
  void upperValueChanged(int upper);

  void lowerPositionChanged(int lower);
  void upperPositionChanged(int upper);

  void sliderPressed(SpanHandle handle);

protected:
  void keyPressEvent    (QKeyEvent*   event) override;
  void mousePressEvent  (QMouseEvent* event) override;
  void mouseMoveEvent   (QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void paintEvent       (QPaintEvent* event) override;

private:
  friend class QxtSpanSliderPrivate;

  private_interface<QxtSpanSlider, QxtSpanSliderPrivate> qxt_d;
};

class QxtSpanSliderPrivate : public QObject, public QxtPrivate<QxtSpanSlider>
{
  Q_OBJECT

public:
  friend class QxtSpanSlider;

    QxtSpanSliderPrivate();
  void initStyleOption(QStyleOptionSlider* option, QxtSpanSlider::SpanHandle handle = QxtSpanSlider::UpperHandle) const;
  int pick(const QPoint& pt) const
  {
    return qxt_p().orientation() == Qt::Horizontal ? pt.x() : pt.y();
  }
  int pixelPosToRangeValue(int pos) const;
  void handleMousePress(const QPoint& pos, QStyle::SubControl& control, int value, QxtSpanSlider::SpanHandle handle);
  void drawHandle(QStylePainter* painter, QxtSpanSlider::SpanHandle handle) const;
  void setupPainter(QPainter* painter, Qt::Orientation orientation, qreal x1, qreal y1, qreal x2, qreal y2) const;
  void drawSpan(QStylePainter* painter, const QRect& rect) const;
  void triggerAction(QAbstractSlider::SliderAction action, bool main);
  void swapControls();

  int lower;
  int upper;
  int lowerPos;
  int upperPos;
  int offset;
  int position;
  QxtSpanSlider::SpanHandle lastPressed;
  QxtSpanSlider::SpanHandle mainControl;
  QStyle::SubControl lowerPressed;
  QStyle::SubControl upperPressed;
  QxtSpanSlider::HandleMovementMode movement;
  bool firstMovement;
  bool blockTracking;

  public Q_SLOTS:
  void updateRange(int min, int max);
  void movePressedHandle();
};

#endif
