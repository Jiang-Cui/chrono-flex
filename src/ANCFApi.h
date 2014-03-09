#ifndef ANCFAPI_H
#define ANCFAPI_H


// Export/import macros

#if (((defined WIN32)|| (defined WIN64))  || (defined(__MINGW32__) || defined(__CYGWIN__)))
		#define IBeamsEXPORT __declspec(dllexport)
		#define IBEamsIMPORT __declspec(dllimport)
#else
		#define IBeamsEXPORT  
		#define IBeamsIMPORT  
#endif


#if defined(IBEAMS_COMPILE_LIBRARY)
#define IBeamsApi IBeamsEXPORT
#else
#define IBeamsApi IBeamsIMPORT
#endif


#endif