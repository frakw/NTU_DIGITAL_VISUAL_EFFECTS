#include "texture.h"

unsigned int TextureFromMat(const Mat& mat)
{

	double min, max;
	cv::minMaxLoc(mat, &min, &max);
	cv::Mat img = mat;
	glPixelStorei(GL_UNPACK_ALIGNMENT, (img.step & 3) ? 1 : 4);


	glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(img.step / img.elemSize()));

	GLenum internalformat = GL_RGB32F;
	if (img.channels() == 4) internalformat = GL_RGBA;
	if (img.channels() == 3) internalformat = GL_RGB;
	if (img.channels() == 2) internalformat = GL_RG;
	if (img.channels() == 1) internalformat = GL_RED;

	GLenum externalformat = GL_BGR;
	if (img.channels() == 1) externalformat = GL_RED; 

	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);


	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	try {
		//internalformat = internals[k];
		glTexImage2D(GL_TEXTURE_2D,
			/* level */				0,
			/* internalFormat */	internalformat,
			/* width */				img.cols,
			/* height */			img.rows,
			/* border */			0,
			/* format */			externalformat,
			/* type */				GL_UNSIGNED_BYTE,
			/* *data */				img.ptr());

		//glGenerateMipmap(GL_TEXTURE_2D);
	}
	catch (std::exception& e)
	{
		printf("%s.\n", e.what());
	}

	return texture;
}

//unsigned int TextureFromFile(const char* path, bool gamma)
//{
//    //string filename = string(path);
//    //filename = directory + '/' + filename;
//
//    unsigned int textureID;
//    glGenTextures(1, &textureID);
//    int width, height, nrComponents;
//    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
//
//    if (data)
//    {
//        GLenum format;
//        if (nrComponents == 1)
//            format = GL_RED;
//        else if (nrComponents == 3)
//            format = GL_RGB;
//        else if (nrComponents == 4)
//            format = GL_RGBA;
//
//        glBindTexture(GL_TEXTURE_2D, textureID);
//        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
//        glGenerateMipmap(GL_TEXTURE_2D);
//
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//
//        stbi_image_free(data);
//    }
//    else
//    {
//        std::cout << "Texture failed to load at path: " << path << std::endl;
//        stbi_image_free(data);
//    }
//
//    return textureID;
//}