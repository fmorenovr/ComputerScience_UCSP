#include <stdio.h>
#include <string>
#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#include <fbxsdk/fbxsdk.h>
#include "objloader.hpp"
#include "../skeleton.h"
#include "../node.h"


using namespace std;
// Very, VERY simple OBJ loader.
// Here is a short list of features a real function would provide : 
// - Binary files. Reading a model should be just a few memcpy's away, not parsing a file at runtime. In short : OBJ is not very great.
// - Animations & bones (includes bones weights)
// - Multiple UVs
// - All attributes should be optional, not "forced"
// - More stable. Change a line in the OBJ file and it crashes.
// - More secure. Change another line and you can inject code.
// - Loading from memory, stream, etc

bool loadOBJ(
	const char * path, 
	std::vector<glm::vec3> & out_vertices, 
	std::vector<glm::vec2> & out_uvs,
	std::vector<glm::vec3> & out_normals
	){
	printf("Loading OBJ file %s...\n", path);

	std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
	std::vector<glm::vec3> temp_vertices; 
	std::vector<glm::vec2> temp_uvs;
	std::vector<glm::vec3> temp_normals;


	FILE * file = fopen(path, "r");
	if( file == NULL ){
		printf("Impossible to open the file ! Are you in the right path ? See Tutorial 1 for details\n");
		getchar();
		return false;
	}

	while( 1 ){

		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		// else : parse lineHeader
		
		if ( strcmp( lineHeader, "v" ) == 0 ){
			glm::vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
			vertex /=50; 
			temp_vertices.push_back(vertex);
		}else if ( strcmp( lineHeader, "vt" ) == 0 ){
			glm::vec2 uv;
			fscanf(file, "%f %f\n", &uv.x, &uv.y );
			uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
			temp_uvs.push_back(uv);
		}else if ( strcmp( lineHeader, "vn" ) == 0 ){
			glm::vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
			temp_normals.push_back(normal);
		}else if ( strcmp( lineHeader, "f" ) == 0 ){
			std::string vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
			if (matches != 9){
				printf("File can't be read by our simple parser :-( Try exporting with other options\n");
				fclose(file);
				return false;
			}
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices    .push_back(uvIndex[0]);
			uvIndices    .push_back(uvIndex[1]);
			uvIndices    .push_back(uvIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);
		}else{
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file);
		}

	}

	// For each vertex of each triangle
	for( unsigned int i=0; i<vertexIndices.size(); i++ ){

		// Get the indices of its attributes
		unsigned int vertexIndex = vertexIndices[i];
		unsigned int uvIndex = uvIndices[i];
		unsigned int normalIndex = normalIndices[i];
		
		// Get the attributes thanks to the index
		glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];
		glm::vec2 uv = temp_uvs[ uvIndex-1 ];
		glm::vec3 normal = temp_normals[ normalIndex-1 ];
		
		// Put the attributes in buffers
		out_vertices.push_back(vertex);
		out_uvs     .push_back(uv);
		out_normals .push_back(normal);

	}
	fclose(file);
	return true;
}

// Get the matrix of the given pose
FbxAMatrix GetPoseMatrix(FbxPose* pPose, int pNodeIndex)
{
    FbxAMatrix lPoseMatrix;
    FbxMatrix lMatrix = pPose->GetMatrix(pNodeIndex);
    memcpy((double*)lPoseMatrix, (double*)lMatrix, sizeof(lMatrix.mData));
    return lPoseMatrix;
}
// Get the geometry offset to a node. It is never inherited by the children.
FbxAMatrix GetGeometry(FbxNode* pNode)
{
    const FbxVector4 lT = pNode->GetGeometricTranslation(FbxNode::eSourcePivot);
    const FbxVector4 lR = pNode->GetGeometricRotation(FbxNode::eSourcePivot);
    const FbxVector4 lS = pNode->GetGeometricScaling(FbxNode::eSourcePivot);
    return FbxAMatrix(lT, lR, lS);
}

FbxAMatrix GetGlobalPosition(FbxNode* pNode, const FbxTime& pTime, FbxPose* pPose, FbxAMatrix* pParentGlobalPosition)
{
    FbxAMatrix lGlobalPosition;
    bool        lPositionFound = false;
    if (pPose)
    {
        int lNodeIndex = pPose->Find(pNode);
        if (lNodeIndex > -1)
        {
            // The bind pose is always a global matrix.
            // If we have a rest pose, we need to check if it is
            // stored in global or local space.
            if (pPose->IsBindPose() || !pPose->IsLocalMatrix(lNodeIndex))
            {
                lGlobalPosition = GetPoseMatrix(pPose, lNodeIndex);
            }
            else
            {
                // We have a local matrix, we need to convert it to
                // a global space matrix.
                FbxAMatrix lParentGlobalPosition;
                if (pParentGlobalPosition)
                {
                    lParentGlobalPosition = *pParentGlobalPosition;
                }
                else
                {
                    if (pNode->GetParent())
                    {
                        lParentGlobalPosition = GetGlobalPosition(pNode->GetParent(), pTime, pPose,NULL);
                    }
                }
                FbxAMatrix lLocalPosition = GetPoseMatrix(pPose, lNodeIndex);
                lGlobalPosition = lParentGlobalPosition * lLocalPosition;
            }
            lPositionFound = true;
        }
    }
    if (!lPositionFound)
    {
        // There is no pose entry for that node, get the current global position instead.
        // Ideally this would use parent global position and local position to compute the global position.
        // Unfortunately the equation 
        //    lGlobalPosition = pParentGlobalPosition * lLocalPosition
        // does not hold when inheritance type is other than "Parent" (RSrs).
        // To compute the parent rotation and scaling is tricky in the RrSs and Rrs cases.
        lGlobalPosition = pNode->EvaluateGlobalTransform(pTime);
    }
    return lGlobalPosition;
}

/**
 * Return a string-based representation based on the attribute type.
 */
FbxString GetAttributeTypeName(FbxNodeAttribute::EType type) { 
	switch(type) { 
		case FbxNodeAttribute::eUnknown: return "unidentified"; 
		case FbxNodeAttribute::eNull: return "null"; 
		case FbxNodeAttribute::eMarker: return "marker"; 
		case FbxNodeAttribute::eSkeleton: return "skeleton"; 
		case FbxNodeAttribute::eMesh: return "mesh"; 
		case FbxNodeAttribute::eNurbs: return "nurbs"; 
		case FbxNodeAttribute::ePatch: return "patch"; 
		case FbxNodeAttribute::eCamera: return "camera"; 
		case FbxNodeAttribute::eCameraStereo: return "stereo"; 
		case FbxNodeAttribute::eCameraSwitcher: return "camera switcher"; 
		case FbxNodeAttribute::eLight: return "light"; 
		case FbxNodeAttribute::eOpticalReference: return "optical reference"; 
		case FbxNodeAttribute::eOpticalMarker: return "marker"; 
		case FbxNodeAttribute::eNurbsCurve: return "nurbs curve"; 
		case FbxNodeAttribute::eTrimNurbsSurface: return "trim nurbs surface"; 
		case FbxNodeAttribute::eBoundary: return "boundary"; 
		case FbxNodeAttribute::eNurbsSurface: return "nurbs surface"; 
		case FbxNodeAttribute::eShape: return "shape"; 
		case FbxNodeAttribute::eLODGroup: return "lodgroup"; 
		case FbxNodeAttribute::eSubDiv: return "subdiv"; 
		default: return "unknown"; 
	} 
}
/* Tab character ("\t") counter */
int numTabs = 0; 

/**
 * Print the required number of tabs.
 */
void PrintTabs() {
	for(int i = 0; i < numTabs; i++)
		std::cout<<"\t";
}

void PrintAttribute(FbxNodeAttribute* pAttribute) {
	if(!pAttribute) return;

	FbxString typeName = GetAttributeTypeName(pAttribute->GetAttributeType());
	FbxString attrName = pAttribute->GetName();
	PrintTabs();
    // Note: to retrieve the character array of a FbxString, use its Buffer() method.
	std::cout << "<attribute type=" << typeName.Buffer() <<"name="<< attrName.Buffer()<<"/>"<< std::endl;
}

void PrintNode(FbxNode* pNode,std::vector<FbxMesh*>& v, std::vector<std::vector<glm::vec3>>& vt, Node* node,float scale) {

	const char* lSkeletonTypes[] = { "Root", "Limb", "Limb Node", "Effector" };
	
	const char* nodeName = pNode->GetName();
	FbxDouble3 translation = pNode->LclTranslation.Get(); 
	FbxDouble3 rotation = pNode->LclRotation.Get(); 
	FbxDouble3 scaling = pNode->LclScaling.Get();

   
	
	numTabs++;

    // Print the node's attributes.
	for(int i = 0; i < pNode->GetNodeAttributeCount(); i++){
		PrintAttribute(pNode->GetNodeAttributeByIndex(i));
		if( pNode->GetNodeAttributeByIndex(i)->GetAttributeType()==FbxNodeAttribute::eMesh){
			v.push_back(pNode->GetMesh());


			std::vector<glm::vec3> vec;

			glm::vec3 gtranslation;
			gtranslation.x=translation[0];
			gtranslation.y=translation[1];
			gtranslation.z=translation[2];

			glm::vec3 grotation;
			grotation.x=rotation[0];
			grotation.y=rotation[1];
			grotation.z=rotation[2];

			glm::vec3 gscaling;
			gscaling.x=scaling[0];
			gscaling.y=scaling[1];
			gscaling.z=scaling[2];

			vec.push_back(gtranslation);
			vec.push_back(grotation);
			vec.push_back(gscaling);

			vt.push_back(vec);

		}else if( pNode->GetNodeAttributeByIndex(i)->GetAttributeType()==FbxNodeAttribute::eSkeleton){
			FbxSkeleton* lSkeleton= pNode->GetSkeleton();
			
			FbxAMatrix fbxam = GetGlobalPosition(pNode,0,NULL,NULL);
			translation = fbxam.GetT ();
			rotation = fbxam.GetR ();
			scaling = fbxam.GetS ();

			glm::vec3 gtranslation;
			gtranslation.x=translation[0];
			gtranslation.y=translation[1];
			gtranslation.z=translation[2];



			glm::vec3 grotation;
			grotation.x=rotation[0];
			grotation.y=rotation[1];
			grotation.z=rotation[2];

			glm::vec3 gscaling;
			gscaling.x=scaling[0];
			gscaling.y=scaling[1];
			gscaling.z=scaling[2];


			glm::mat4 ModelMatrix = glm::mat4(1.0);
			ModelMatrix = glm::translate(ModelMatrix,gtranslation/scale);	
			ModelMatrix = glm::rotate(ModelMatrix,glm::radians(grotation.x),glm::vec3(1.0f,0.0f,0.0f));
			ModelMatrix = glm::rotate(ModelMatrix,glm::radians(grotation.y),glm::vec3(0.0f,1.0f,0.0f));
			ModelMatrix = glm::rotate(ModelMatrix,glm::radians(grotation.z),glm::vec3(0.0f,0.0f,1.0f));
			ModelMatrix = glm::scale(ModelMatrix,gscaling);	

			Node* next;
			if(node->label=="init")
			{
				node->label = std::string(nodeName);
				node->ModelMatrix = glm::mat4(ModelMatrix);
				next=node;

			}else{
				next=new Node(node,std::string(nodeName));
				next->ModelMatrix = glm::mat4(ModelMatrix);
				node->addChild(next);
				node=next;
			}
		}
			
	}

    // Recursively print the children.
	for(int j = 0; j < pNode->GetChildCount(); j++)
		PrintNode(pNode->GetChild(j),v,vt,node,scale);

	numTabs--;
}


bool loadFBX(
	const char * path, 
	std::vector<glm::vec3> & out_vertices, 
	std::vector<glm::vec2> & out_uvs,
	std::vector<glm::vec3> & out_normals,
	Skeleton* skel,
	float scale
	){

      // Change the following filename to a suitable filename value.
    // Initialize the SDK manager. This object handles memory management.
	FbxManager* lSdkManager = FbxManager::Create();

	FbxIOSettings *ios = FbxIOSettings::Create(lSdkManager, IOSROOT);
	lSdkManager->SetIOSettings(ios);

    // Create an importer using the SDK manager.
	FbxImporter* lImporter = FbxImporter::Create(lSdkManager,"");

    // Use the first argument as the filename for the importer.
	bool lImportStatus = lImporter->Initialize(path, -1, lSdkManager->GetIOSettings());
	if(!lImportStatus) {
		printf("Call to FbxImporter::Initialize() failed.\n"); 
		printf("Error returned: %s\n\n", lImporter->GetStatus().GetErrorString()); 
		exit(-1);
	}


	FbxScene* lScene = FbxScene::Create(lSdkManager,"myScene");

    // Import the contents of the file into the scene.
	lImporter->Import(lScene);

    // The file is imported, so get rid of the importer.
	lImporter->Destroy();

	FbxNode* lRootNode = lScene->GetRootNode();
	std::vector<FbxMesh*> vmesh;
	std::vector<std::vector<glm::vec3>> vmeshTransform;
	Node* root = skel->root;

	if(lRootNode) {
		for(int i = 0; i < lRootNode->GetChildCount(); i++)
			PrintNode(lRootNode->GetChild(i),vmesh,vmeshTransform,root,scale);
	}
	std::cout << "find " << vmesh.size() << " meshes" << std::endl;

	for(int meshId = 1;meshId < 2;meshId++){


		std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
		std::vector<glm::vec3> temp_vertices; 
		std::vector<glm::vec2> temp_uvs;
		std::vector<glm::vec3> temp_normals;


		FbxArray< FbxVector4 > pNormals;
		vmesh.at(meshId)->GetPolygonVertexNormals (pNormals);
		for(int i=0;i < pNormals.Size();i++){
			glm::vec3 n;
			n.x = pNormals [i][0];
			n.y = pNormals [i][1];
			n.z = pNormals [i][2];
			temp_normals.push_back(n);
		}

		int cont=0;
		for(int i=0;i < vmesh.at(meshId)->GetPolygonCount();i++){
			if(vmesh.at(meshId)->GetPolygonSize(i)==3)
			{
				for(int j=0;j<3;j++){
					vertexIndices.push_back(vmesh.at(meshId)->GetPolygonVertex(i,j));
					int v_index = vmesh.at(meshId)->GetPolygonVertex(i,j);

					out_normals.push_back(temp_normals[cont]);
					cont ++;

				}
			}
			else if(vmesh.at(meshId)->GetPolygonSize(i)==4)
			{
				int a = vmesh.at(meshId)->GetPolygonVertex(i,0);
				int b = vmesh.at(meshId)->GetPolygonVertex(i,1);
				int c = vmesh.at(meshId)->GetPolygonVertex(i,2);
				int d = vmesh.at(meshId)->GetPolygonVertex(i,3);

				vertexIndices.push_back(a);
				vertexIndices.push_back(b);
				vertexIndices.push_back(c);
				vertexIndices.push_back(a);
				vertexIndices.push_back(c);
				vertexIndices.push_back(d);

				out_normals.push_back(temp_normals[cont]);
				out_normals.push_back(temp_normals[cont+1]);
				out_normals.push_back(temp_normals[cont+2]);
				out_normals.push_back(temp_normals[cont]);
				out_normals.push_back(temp_normals[cont+2]);
				out_normals.push_back(temp_normals[cont+3]);
				cont +=4;




			}else if(vmesh.at(meshId)->GetPolygonSize(i)==5){

				int a = vmesh.at(meshId)->GetPolygonVertex(i,0);
				int b = vmesh.at(meshId)->GetPolygonVertex(i,1);
				int c = vmesh.at(meshId)->GetPolygonVertex(i,2);
				int d = vmesh.at(meshId)->GetPolygonVertex(i,3);
				int e = vmesh.at(meshId)->GetPolygonVertex(i,4);

				vertexIndices.push_back(a);
				vertexIndices.push_back(b);
				vertexIndices.push_back(c);
				vertexIndices.push_back(a);
				vertexIndices.push_back(c);
				vertexIndices.push_back(d);
				vertexIndices.push_back(a);
				vertexIndices.push_back(d);
				vertexIndices.push_back(e);


				out_normals.push_back(temp_normals[cont]);
				out_normals.push_back(temp_normals[cont+1]);
				out_normals.push_back(temp_normals[cont+2]);
				out_normals.push_back(temp_normals[cont]);
				out_normals.push_back(temp_normals[cont+2]);
				out_normals.push_back(temp_normals[cont+3]);
				out_normals.push_back(temp_normals[cont]);
				out_normals.push_back(temp_normals[cont+3]);
				out_normals.push_back(temp_normals[cont+4]);
				cont +=5;

			}
		}


		FbxVector4* vs = vmesh.at(meshId)->GetControlPoints();
		for(int i=0;i < vmesh.at(meshId)->GetControlPointsCount();i++){
			glm::vec4 vertex;
			vertex.x = vs [i][0];
			vertex.y = vs [i][1];
			vertex.z = vs [i][2];
			vertex.w = 1;
			glm::mat4 ModelMatrix = glm::mat4(1.0);

			ModelMatrix = glm::translate(ModelMatrix,vmeshTransform.at(meshId).at(0));		
			ModelMatrix = glm::rotate(ModelMatrix,glm::radians(vmeshTransform.at(meshId).at(1).x),glm::vec3(1.0f,0.0f,0.0f));
			ModelMatrix = glm::rotate(ModelMatrix,glm::radians(vmeshTransform.at(meshId).at(1).y),glm::vec3(0.0f,1.0f,0.0f));
			ModelMatrix = glm::rotate(ModelMatrix,glm::radians(vmeshTransform.at(meshId).at(1).z),glm::vec3(0.0f,0.0f,1.0f));
			ModelMatrix = glm::scale(ModelMatrix,vmeshTransform.at(meshId).at(2));	

			vertex = /*ModelMatrix***/vertex;
			vertex /= scale;
			//cout << vertex.x;
			temp_vertices.push_back(glm::vec3(vertex.x,vertex.y,vertex.z));

		}


		FbxArray< FbxVector2 > pUVs;
		vmesh.at(meshId)->GetPolygonVertexUVs ("",pUVs);
		for(int i=0;i < pUVs.Size();i++){
			glm::vec2 uv;
			uv.x = pUVs [i][0];
			temp_uvs.push_back(uv);
		}




		// For each vertex of each triangle
		for( unsigned int i=0; i<vertexIndices.size(); i++ ){

		// Get the indices of its attributes
			unsigned int vertexIndex = vertexIndices[i];
		// Get the attributes thanks to the index
			glm::vec3 vertex = temp_vertices[ vertexIndex];
		//glm::vec2 uv = vmesh.at(meshId)->[ uvIndex-1
		//glm::vec3 normal = temp_normals[ i];
		//out_normals.push_back(normal); 

		// Put the attributes in buffers
			out_vertices.push_back(vertex);
		//out_uvs     .push_back(uv);

		}
		cout << "Mesh n°" << meshId << " count " << out_vertices.size()<<endl;
	} 

    // Destroy the SDK manager and all the other objects it was handling.
	lSdkManager->Destroy();
}

#ifdef USE_ASSIMP // don't use this #define, it's only for me (it AssImp fails to compile on your machine, at least all the other tutorials still work)

// Include AssImp
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

bool loadAssImp(
	const char * path, 
	std::vector<unsigned short> & indices,
	std::vector<glm::vec3> & vertices,
	std::vector<glm::vec2> & uvs,
	std::vector<glm::vec3> & normals
	){

	Assimp::Importer importer;

	const aiScene* scene = importer.ReadFile(path, 0/*aiProcess_JoinIdenticalVertices | aiProcess_SortByPType*/);
	if( !scene) {
		fprintf( stderr, importer.GetErrorString());
		getchar();
		return false;
	}
	const aiMesh* mesh = scene->mMeshes[0]; // In this simple example code we always use the 1rst mesh (in OBJ files there is often only one anyway)

	// Fill vertices positions
	vertices.reserve(mesh->mNumVertices);
	for(unsigned int i=0; i<mesh->mNumVertices; i++){
		aiVector3D pos = mesh->mVertices[i];
		vertices.push_back(glm::vec3(pos.x, pos.y, pos.z));
	}

	// Fill vertices texture coordinates
	uvs.reserve(mesh->mNumVertices);
	for(unsigned int i=0; i<mesh->mNumVertices; i++){
		aiVector3D UVW = mesh->mTextureCoords[0][i]; // Assume only 1 set of UV coords; AssImp supports 8 UV sets.
		uvs.push_back(glm::vec2(UVW.x, UVW.y));
	}

	// Fill vertices normals
	normals.reserve(mesh->mNumVertices);
	for(unsigned int i=0; i<mesh->mNumVertices; i++){
		aiVector3D n = mesh->mNormals[i];
		normals.push_back(glm::vec3(n.x, n.y, n.z));
	}


	// Fill face indices
	indices.reserve(3*mesh->mNumFaces);
	for (unsigned int i=0; i<mesh->mNumFaces; i++){
		// Assume the model has only triangles.
		indices.push_back(mesh->mFaces[i].mIndices[0]);
		indices.push_back(mesh->mFaces[i].mIndices[1]);
		indices.push_back(mesh->mFaces[i].mIndices[2]);
	}
	
	// The "scene" pointer will be deleted automatically by "importer"
	return true;
}

#endif
