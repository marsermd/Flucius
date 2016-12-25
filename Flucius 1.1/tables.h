#ifndef TABLES_H
#define TABLES_H
#include <GL\glew.h>

//Vertices on the middle of cube's edges
const GLfloat vertexPositionTable[12][3] = {
	{0.0f, 0.0f, 0.5f},
	{0.0f, 0.5f, 1.0f},
	{0.0f, 1.0f, 0.5f},
	{0.0f, 0.5f, 0.0f},
	{1.0f, 0.0f, 0.5f},
	{1.0f, 0.5f, 1.0f},
	{1.0f, 1.0f, 0.5f},
	{1.0f, 0.5f, 0.0f},
	{0.5f, 0.0f, 0.0f},
	{0.5f, 0.0f, 1.0f},
	{0.5f, 1.0f, 1.0f},
	{0.5f, 1.0f, 0.0f}};

//maps egde index to pair of alligned vertices
const GLuint verticesForEdgesTable[12][2] = {	
	{0,	1},
	{1,	2},
	{2,	3},
	{3,	0},
	{4,	5},
	{5,	6},
	{6,	7},
	{7,	4},
	{0,	4},
	{1,	5},
	{2,	6},
	{3,	7}};

//maps 8-bit cube type id to 12-bit edge bitprofile
//cube type id depends on enabled/disabled corner vertices
//currently unused.
const GLuint edgesTable[256] = {
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   
};

//maps cube index to triangles to draw
#define X 255
const GLuint trianglesTable[256][16] =
{
	{X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X},		//0
	{0, 8, 3, X, X, X, X, X, X, X, X, X, X, X, X, X},		//1
	{0, 1, 9, X, X, X, X, X, X, X, X, X, X, X, X, X},		//2
	{1, 8, 3, 9, 8, 1, X, X, X, X, X, X, X, X, X, X},		//3
	{1, 2, 10, X, X, X, X, X, X, X, X, X, X, X, X, X},		//4
	{0, 8, 3, 1, 2, 10, X, X, X, X, X, X, X, X, X, X},		//5
	{9, 2, 10, 0, 2, 9, X, X, X, X, X, X, X, X, X, X},		//6
	{2, 8, 3, 2, 10, 8, 10, 9, 8, X, X, X, X, X, X, X},		//7
	{3, 11, 2, X, X, X, X, X, X, X, X, X, X, X, X, X},		//8
	{0, 11, 2, 8, 11, 0, X, X, X, X, X, X, X, X, X, X},		//9
	{1, 9, 0, 2, 3, 11, X, X, X, X, X, X, X, X, X, X},		//10
	{1, 11, 2, 1, 9, 11, 9, 8, 11, X, X, X, X, X, X, X},	//11
	{3, 10, 1, 11, 10, 3, X, X, X, X, X, X, X, X, X, X},	//12
	{0, 10, 1, 0, 8, 10, 8, 11, 10, X, X, X, X, X, X, X},	//13
	{3, 9, 0, 3, 11, 9, 11, 10, 9, X, X, X, X, X, X, X},	//14
	{9, 8, 10, 10, 8, 11, X, X, X, X, X, X, X, X, X, X},	//15
	{4, 7, 8, X, X, X, X, X, X, X, X, X, X, X, X, X},		//16
	{4, 3, 0, 7, 3, 4, X, X, X, X, X, X, X, X, X, X},		//17
	{0, 1, 9, 8, 4, 7, X, X, X, X, X, X, X, X, X, X},		//18
	{4, 1, 9, 4, 7, 1, 7, 3, 1, X, X, X, X, X, X, X},		//19
	{1, 2, 10, 8, 4, 7, X, X, X, X, X, X, X, X, X, X},		//20
	{3, 4, 7, 3, 0, 4, 1, 2, 10, X, X, X, X, X, X, X},		//21
	{9, 2, 10, 9, 0, 2, 8, 4, 7, X, X, X, X, X, X, X},		//22
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, X, X, X, X},		//23
	{8, 4, 7, 3, 11, 2, X, X, X, X, X, X, X, X, X, X},		//24
	{11, 4, 7, 11, 2, 4, 2, 0, 4, X, X, X, X, X, X, X},		//25
	{9, 0, 1, 8, 4, 7, 2, 3, 11, X, X, X, X, X, X, X},		//26
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, X, X, X, X},	//27
	{3, 10, 1, 3, 11, 10, 7, 8, 4, X, X, X, X, X, X, X},	//28
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, X, X, X, X},	//29
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, X, X, X, X},	//30
	{4, 7, 11, 4, 11, 9, 9, 11, 10, X, X, X, X, X, X, X},	//31
	{9, 5, 4, X, X, X, X, X, X, X, X, X, X, X, X, X},		//32
	{9, 5, 4, 0, 8, 3, X, X, X, X, X, X, X, X, X, X},		//33
	{0, 5, 4, 1, 5, 0, X, X, X, X, X, X, X, X, X, X},		//34
	{8, 5, 4, 8, 3, 5, 3, 1, 5, X, X, X, X, X, X, X},		//35
	{1, 2, 10, 9, 5, 4, X, X, X, X, X, X, X, X, X, X},		//36
	{3, 0, 8, 1, 2, 10, 4, 9, 5, X, X, X, X, X, X, X},		//37
	{5, 2, 10, 5, 4, 2, 4, 0, 2, X, X, X, X, X, X, X},		//38
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, X, X, X, X},		//39
	{9, 5, 4, 2, 3, 11, X, X, X, X, X, X, X, X, X, X},		//40
	{0, 11, 2, 0, 8, 11, 4, 9, 5, X, X, X, X, X, X, X},		//41
	{0, 5, 4, 0, 1, 5, 2, 3, 11, X, X, X, X, X, X, X},		//42
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, X, X, X, X},		//43
	{10, 3, 11, 10, 1, 3, 9, 5, 4, X, X, X, X, X, X, X},	//44
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, X, X, X, X},	//45
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, X, X, X, X},	//46
	{5, 4, 8, 5, 8, 10, 10, 8, 11, X, X, X, X, X, X, X},	//47
	{9, 7, 8, 5, 7, 9, X, X, X, X, X, X, X, X, X, X},		//48
	{9, 3, 0, 9, 5, 3, 5, 7, 3, X, X, X, X, X, X, X},		//49
	{0, 7, 8, 0, 1, 7, 1, 5, 7, X, X, X, X, X, X, X},		//50
	{1, 5, 3, 3, 5, 7, X, X, X, X, X, X, X, X, X, X},		//51
	{9, 7, 8, 9, 5, 7, 10, 1, 2, X, X, X, X, X, X, X},		//52
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, X, X, X, X},		//53
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, X, X, X, X},		//54
	{2, 10, 5, 2, 5, 3, 3, 5, 7, X, X, X, X, X, X, X},		//55
	{7, 9, 5, 7, 8, 9, 3, 11, 2, X, X, X, X, X, X, X},		//56
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, X, X, X, X},		//57
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, X, X, X, X},		//58
	{11, 2, 1, 11, 1, 7, 7, 1, 5, X, X, X, X, X, X, X},		//59
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, X, X, X, X},	//60
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, X},	//61
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, X},	//62
	{11, 10, 5, 7, 11, 5, X, X, X, X, X, X, X, X, X, X},	//63
	{10, 6, 5, X, X, X, X, X, X, X, X, X, X, X, X, X},		//64
	{0, 8, 3, 5, 10, 6, X, X, X, X, X, X, X, X, X, X},		//65
	{9, 0, 1, 5, 10, 6, X, X, X, X, X, X, X, X, X, X},		//66
	{1, 8, 3, 1, 9, 8, 5, 10, 6, X, X, X, X, X, X, X},		//67
	{1, 6, 5, 2, 6, 1, X, X, X, X, X, X, X, X, X, X},		//68
	{1, 6, 5, 1, 2, 6, 3, 0, 8, X, X, X, X, X, X, X},		//69
	{9, 6, 5, 9, 0, 6, 0, 2, 6, X, X, X, X, X, X, X},		//70
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, X, X, X, X},		//71
	{2, 3, 11, 10, 6, 5, X, X, X, X, X, X, X, X, X, X},		//72
	{11, 0, 8, 11, 2, 0, 10, 6, 5, X, X, X, X, X, X, X},	//73
	{0, 1, 9, 2, 3, 11, 5, 10, 6, X, X, X, X, X, X, X},		//74
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, X, X, X, X},	//75
	{6, 3, 11, 6, 5, 3, 5, 1, 3, X, X, X, X, X, X, X},		//76
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, X, X, X, X},	//77
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, X, X, X, X},		//78
	{6, 5, 9, 6, 9, 11, 11, 9, 8, X, X, X, X, X, X, X},		//79
	{5, 10, 6, 4, 7, 8, X, X, X, X, X, X, X, X, X, X},		//80
	{4, 3, 0, 4, 7, 3, 6, 5, 10, X, X, X, X, X, X, X},		//81
	{1, 9, 0, 5, 10, 6, 8, 4, 7, X, X, X, X, X, X, X},		//82
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, X, X, X, X},		//83
	{6, 1, 2, 6, 5, 1, 4, 7, 8, X, X, X, X, X, X, X},		//84
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, X, X, X, X},		//85
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, X, X, X, X},		//86
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, X},		//87
	{3, 11, 2, 7, 8, 4, 10, 6, 5, X, X, X, X, X, X, X},		//88
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, X, X, X, X},		//89
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, X, X, X, X},		//90
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, X},	//91
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, X, X, X, X},		//92
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, X},	//93
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, X},		//94
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, X, X, X, X},		//95
	{10, 4, 9, 6, 4, 10, X, X, X, X, X, X, X, X, X, X},		//96
	{4, 10, 6, 4, 9, 10, 0, 8, 3, X, X, X, X, X, X, X},		//97
	{10, 0, 1, 10, 6, 0, 6, 4, 0, X, X, X, X, X, X, X},		//98
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, X, X, X, X},		//99
	{1, 4, 9, 1, 2, 4, 2, 6, 4, X, X, X, X, X, X, X},		//100
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, X, X, X, X},		//101
	{0, 2, 4, 4, 2, 6, X, X, X, X, X, X, X, X, X, X},		//102
	{8, 3, 2, 8, 2, 4, 4, 2, 6, X, X, X, X, X, X, X},		//103
	{10, 4, 9, 10, 6, 4, 11, 2, 3, X, X, X, X, X, X, X},	//104
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, X, X, X, X},	//105
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, X, X, X, X},		//106
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, X},	//107
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, X, X, X, X},		//108
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, X},		//109
	{3, 11, 6, 3, 6, 0, 0, 6, 4, X, X, X, X, X, X, X},		//110
	{ 6, 4, 8, 11, 6, 8, X, X, X, X, X, X, X, X, X, X },	//111
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, X, X, X, X, X, X, X },	//112
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, X, X, X, X },	//113
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, X, X, X, X },	//114
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, X, X, X, X, X, X, X },	//115
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, X, X, X, X },		//116
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, X },		//117
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, X, X, X, X, X, X, X },		//118
	{ 7, 3, 2, 6, 7, 2, X, X, X, X, X, X, X, X, X, X },		//119
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, X, X, X, X },	//120
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, X },	//121
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, X },	//122
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, X, X, X, X },	//123
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, X },	//124
	{ 0, 9, 1, 11, 6, 7, X, X, X, X, X, X, X, X, X, X },	//125
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, X, X, X, X },	//126
	{ 7, 11, 6, X, X, X, X, X, X, X, X, X, X, X, X, X },	//127
	{ 7, 6, 11, X, X, X, X, X, X, X, X, X, X, X, X, X },	//128
	{ 3, 0, 8, 11, 7, 6, X, X, X, X, X, X, X, X, X, X },	//129
	{ 0, 1, 9, 11, 7, 6, X, X, X, X, X, X, X, X, X, X },	//130
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, X, X, X, X, X, X, X },	//131
	{ 10, 1, 2, 6, 11, 7, X, X, X, X, X, X, X, X, X, X },	//132
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, X, X, X, X, X, X, X },	//133
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, X, X, X, X, X, X, X },	//134
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, X, X, X, X },	//135
	{ 7, 2, 3, 6, 2, 7, X, X, X, X, X, X, X, X, X, X },		//136
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, X, X, X, X, X, X, X },		//137
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, X, X, X, X, X, X, X },		//138
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, X, X, X, X },		//139
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, X, X, X, X, X, X, X },	//140
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, X, X, X, X },	//141
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, X, X, X, X },	//142
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, X, X, X, X, X, X, X },	//143
	{ 6, 8, 4, 11, 8, 6, X, X, X, X, X, X, X, X, X, X },	//144
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, X, X, X, X, X, X, X },	//145
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, X, X, X, X, X, X, X },	//146
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, X, X, X, X },	//147
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, X, X, X, X, X, X, X },	//148
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, X, X, X, X },	//149
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, X, X, X, X },	//150
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, X },	//151
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, X, X, X, X, X, X, X },		//152
	{ 0, 4, 2, 4, 6, 2, X, X, X, X, X, X, X, X, X, X },		//153
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, X, X, X, X },		//154
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, X, X, X, X, X, X, X },		//155
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, X, X, X, X },	//156
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, X, X, X, X, X, X, X },	//157
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, X },	//158
	{ 10, 9, 4, 6, 10, 4, X, X, X, X, X, X, X, X, X, X },	//159
	{ 4, 9, 5, 7, 6, 11, X, X, X, X, X, X, X, X, X, X },	//160
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, X, X, X, X, X, X, X },	//161
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, X, X, X, X, X, X, X },	//162
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, X, X, X, X },	//163
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, X, X, X, X, X, X, X },	//164
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, X, X, X, X },	//165
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, X, X, X, X },	//166
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, X },	//167
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, X, X, X, X, X, X, X },		//168
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, X, X, X, X },		//169
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, X, X, X, X },		//170
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, X },		//171
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, X, X, X, X },	//172
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, X },	//173
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, X },//174
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, X, X, X, X },	//175
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, X, X, X, X, X, X, X },	//176
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, X, X, X, X },	//177
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, X, X, X, X },	//178
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, X, X, X, X, X, X, X },	//179
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, X, X, X, X },	//180
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, X },	//181
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, X },	//182
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, X, X, X, X },	//183
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, X, X, X, X },		//184
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, X, X, X, X, X, X, X },		//185
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, X },		//186
	{ 1, 5, 6, 2, 1, 6, X, X, X, X, X, X, X, X, X, X },		//187
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, X },	//188
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, X, X, X, X },	//189
	{ 0, 3, 8, 5, 6, 10, X, X, X, X, X, X, X, X, X, X },	//190
	{ 10, 5, 6, X, X, X, X, X, X, X, X, X, X, X, X, X },	//191
	{ 11, 5, 10, 7, 5, 11, X, X, X, X, X, X, X, X, X, X },	//192
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, X, X, X, X, X, X, X },	//193
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, X, X, X, X, X, X, X },	//194
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, X, X, X, X },	//195
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, X, X, X, X, X, X, X },	//196
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, X, X, X, X },	//197
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, X, X, X, X },	//198
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, X },	//199
	{2, 5, 10, 2, 3, 5, 3, 7, 5, X, X, X, X, X, X, X},		//200
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, X, X, X, X},		//201
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, X, X, X, X},		//202
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, X},		//203
	{1, 3, 5, 3, 7, 5, X, X, X, X, X, X, X, X, X, X},		//204
	{0, 8, 7, 0, 7, 1, 1, 7, 5, X, X, X, X, X, X, X},		//205
	{9, 0, 3, 9, 3, 5, 5, 3, 7, X, X, X, X, X, X, X},		//206
	{9, 8, 7, 5, 9, 7, X, X, X, X, X, X, X, X, X, X},		//207
	{5, 8, 4, 5, 10, 8, 10, 11, 8, X, X, X, X, X, X, X},	//208
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, X, X, X, X},	//209
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, X, X, X, X},	//210
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, X},	//211
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, X, X, X, X},		//212
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, X},	//213
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, X},		//214
	{9, 4, 5, 2, 11, 3, X, X, X, X, X, X, X, X, X, X},		//215
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, X, X, X, X},		//216
	{5, 10, 2, 5, 2, 4, 4, 2, 0, X, X, X, X, X, X, X},		//217
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, X},		//218
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, X, X, X, X},		//219
	{8, 4, 5, 8, 5, 3, 3, 5, 1, X, X, X, X, X, X, X},		//220
	{0, 4, 5, 1, 0, 5, X, X, X, X, X, X, X, X, X, X},		//221
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, X, X, X, X},		//222
	{9, 4, 5, X, X, X, X, X, X, X, X, X, X, X, X, X},		//223
	{4, 11, 7, 4, 9, 11, 9, 10, 11, X, X, X, X, X, X, X},	//224
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, X, X, X, X},	//225
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, X, X, X, X},	//226
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, X},	//227
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, X, X, X, X},	//228
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, X},	//229
	{11, 7, 4, 11, 4, 2, 2, 4, 0, X, X, X, X, X, X, X},		//230
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, X, X, X, X},		//231
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, X, X, X, X},		//232
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, X},		//233
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, X},	//234
	{1, 10, 2, 8, 7, 4, X, X, X, X, X, X, X, X, X, X},		//235
	{4, 9, 1, 4, 1, 7, 7, 1, 3, X, X, X, X, X, X, X},		//236
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, X, X, X, X},		//237
	{4, 0, 3, 7, 4, 3, X, X, X, X, X, X, X, X, X, X},		//238
	{4, 8, 7, X, X, X, X, X, X, X, X, X, X, X, X, X},		//239
	{9, 10, 8, 10, 11, 8, X, X, X, X, X, X, X, X, X, X},	//240
	{3, 0, 9, 3, 9, 11, 11, 9, 10, X, X, X, X, X, X, X},	//241
	{0, 1, 10, 0, 10, 8, 8, 10, 11, X, X, X, X, X, X, X},	//242
	{3, 1, 10, 11, 3, 10, X, X, X, X, X, X, X, X, X, X},	//243
	{1, 2, 11, 1, 11, 9, 9, 11, 8, X, X, X, X, X, X, X},	//244
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, X, X, X, X},		//245
	{0, 2, 11, 8, 0, 11, X, X, X, X, X, X, X, X, X, X},		//246
	{3, 2, 11, X, X, X, X, X, X, X, X, X, X, X, X, X},		//247
	{2, 3, 8, 2, 8, 10, 10, 8, 9, X, X, X, X, X, X, X},		//248
	{9, 10, 2, 0, 9, 2, X, X, X, X, X, X, X, X, X, X},		//249
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, X, X, X, X},		//250
	{1, 10, 2, X, X, X, X, X, X, X, X, X, X, X, X, X},		//251
	{1, 3, 8, 9, 1, 8, X, X, X, X, X, X, X, X, X, X},		//252
	{0, 9, 1, X, X, X, X, X, X, X, X, X, X, X, X, X},		//253
	{0, 3, 8, X, X, X, X, X, X, X, X, X, X, X, X, X},		//254
	{X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X}		//255
};
#undef X


//maps cube index to vertices count
const GLuint vertexCountTable[256] =
{
	0,
	3,
	3,
	6,
	3,
	6,
	6,
	9,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	6,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	6,
	9,
	9,
	6,
	9,
	12,
	12,
	9,
	9,
	12,
	12,
	9,
	12,
	15,
	15,
	6,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	12,
	12,
	15,
	12,
	15,
	15,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	6,
	9,
	9,
	12,
	12,
	15,
	12,
	15,
	9,
	6,
	9,
	12,
	12,
	9,
	12,
	15,
	9,
	6,
	12,
	15,
	15,
	12,
	15,
	6,
	12,
	3,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	6,
	12,
	9,
	12,
	9,
	15,
	6,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	12,
	12,
	15,
	12,
	15,
	15,
	12,
	9,
	12,
	12,
	9,
	12,
	15,
	15,
	12,
	12,
	9,
	15,
	6,
	15,
	12,
	6,
	3,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	12,
	12,
	15,
	6,
	9,
	9,
	6,
	9,
	12,
	12,
	15,
	12,
	15,
	15,
	6,
	12,
	9,
	15,
	12,
	9,
	6,
	12,
	3,
	9,
	12,
	12,
	15,
	12,
	15,
	9,
	12,
	12,
	15,
	15,
	6,
	9,
	12,
	6,
	3,
	6,
	9,
	9,
	6,
	9,
	12,
	6,
	3,
	9,
	6,
	12,
	3,
	6,
	3,
	3,
	0,
};
#endif