#ifndef QXTGLOBAL_H
/****************************************************************************
** Copyright (c) 2006 - 2011, the LibQxt project.
** See the Qxt AUTHORS file for a list of authors and copyright holders.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
**     * Redistributions of source code must retain the above copyright
**       notice, this list of conditions and the following disclaimer.
**     * Redistributions in binary form must reproduce the above copyright
**       notice, this list of conditions and the following disclaimer in the
**       documentation and/or other materials provided with the distribution.
**     * Neither the name of the LibQxt project nor the
**       names of its contributors may be used to endorse or promote products
**       derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
** ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
** WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
** DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
** (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
** LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
** ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**
** <http://libqxt.org>  <foundation@libqxt.org>
*****************************************************************************/

#define QXTGLOBAL_H

#include <QtGlobal>

#define QXT_DECLARE_PRIVATE(PUB) friend class PUB##Private; QxtPrivateInterface<PUB, PUB##Private> qxt_d;
#define QXT_DECLARE_PUBLIC(PUB) friend class PUB;
#define QXT_INIT_PRIVATE(PUB) qxt_d.setPublic(this);
#define QXT_D(PUB) PUB##Private& d = qxt_d()
#define QXT_P(PUB) PUB& p = qxt_p()

template <typename PUB>
class QxtPrivate
{
public:
    virtual ~QxtPrivate()
    {}
    inline void QXT_setPublic(PUB* pub)
    {
        qxt_p_ptr = pub;
    }

protected:
    inline PUB& qxt_p()
    {
        return *qxt_p_ptr;
    }
    inline const PUB& qxt_p() const
    {
        return *qxt_p_ptr;
    }
    inline PUB* qxt_ptr()
    {
        return qxt_p_ptr;
    }
    inline const PUB* qxt_ptr() const
    {
        return qxt_p_ptr;
    }

private:
    PUB* qxt_p_ptr;
};

template <typename PUB, typename PVT>
class QxtPrivateInterface
{
    friend class QxtPrivate<PUB>;
public:
    QxtPrivateInterface()
    {
        pvt = new PVT;
    }
    ~QxtPrivateInterface()
    {
        delete pvt;
    }

    inline void setPublic(PUB* pub)
    {
        pvt->QXT_setPublic(pub);
    }
    inline PVT& operator()()
    {
        return *static_cast<PVT*>(pvt);
    }
    inline const PVT& operator()() const
    {
        return *static_cast<PVT*>(pvt);
    }
    inline PVT * operator->()
    {
	return static_cast<PVT*>(pvt);
    }
    inline const PVT * operator->() const
    {
	return static_cast<PVT*>(pvt);
    }
private:
    QxtPrivateInterface(const QxtPrivateInterface&) { }
    QxtPrivateInterface& operator=(const QxtPrivateInterface&) { }
    QxtPrivate<PUB>* pvt;
};

#endif // QXT_GLOBAL
