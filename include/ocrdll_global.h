#ifndef OCRDLL_GLOBAL_H
#define OCRDLL_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(OCRDLL_LIBRARY)
#  define OCRDLL_EXPORT Q_DECL_EXPORT
#else
#  define OCRDLL_EXPORT Q_DECL_IMPORT
#endif

#endif // OCRDLL_GLOBAL_H
