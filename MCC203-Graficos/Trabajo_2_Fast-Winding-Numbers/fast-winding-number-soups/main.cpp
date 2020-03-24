#include "utils/solidAngle.h"
#include "utils/windingVisualize.h"

std::string outputFile = "../windingNumbers/windings.dmat";
std::string outApproxFile = "../windingNumbers/approxWindings.dmat";
std::string inputFile = "../data/bunny.mesh";
std::string queryFile = "../data/query.dmat";

int main(int argc, char *argv[]){
  std::cout<<"USAGE:\n./fastwinding input.[mesh|msh|obj|off|ply|stl|wrl] query.dmat \nContains a 3D triangle mesh query.dmat  contains a #Q by 3 matrix of input query points output.dmat will contain a #Q by 1 vector of output winding numbers\n\n";

  /* =================================
            Reading Mesh
  ===================================*/

  // Load mesh: (V,T) tet-mesh of convex hull, F contains facets of input
  // surface mesh _after_ self-intersection resolution
  igl::opengl::glfw::Viewer vMesh;
  igl::readMESH(inputFile,V,T,F);
  vMesh.data().set_mesh(V, F);
  vMesh.launch();

  /* =================================
            Barycenters
  ===================================*/

  // Compute barycenters of all tets
  igl::barycenter(V,T,BC);

  /* =================================
            Winding Numbers
  ===================================*/

  // Compute generalized winding number at all barycenters
  cout<<"Computing winding number over all "<<T.rows()<<" tets..."<<endl;
  
  auto startTime = chrono::high_resolution_clock::now();
  
  igl::winding_number(V,F,BC,W);

  // Extract interior tets
  MatrixXi CT((W.array()>0.5).count(),4);
  {
    size_t k = 0;
    for(size_t t = 0;t<T.rows();t++)
    {
      if(W(t)>0.5)
      {
        CT.row(k) = T.row(t);
        k++;
      }
    }
  }
  // find bounary facets of interior tets
  igl::boundary_facets(CT,G);
  // boundary_facets seems to be reversed...
  G = G.rowwise().reverse().eval();

  // normalize
  W = (W.array() - W.minCoeff())/(W.maxCoeff()-W.minCoeff());

  auto endTime = chrono::high_resolution_clock::now();
  auto wTime = chrono::duration_cast<chrono::milliseconds>(endTime-startTime).count();

  cout<<"\n\nComputing winding numbers finished in time "<<wTime<<" ms"<<endl;

  // saving winding numbers
  igl::writeDMAT(outputFile, W, true);
  
  /* =================================
      Approximate Winding Numbers
  ===================================*/

  igl::readDMAT(queryFile, Q);
  Eigen::VectorXf Wapprox(T.rows());
  startTime = chrono::high_resolution_clock::now();
  
  HDK_Sample::UT_SolidAngle<float,float> solid_angle;

  std::vector<HDK_Sample::UT_Vector3T<float> > U(V.rows());
  for(int i = 0;i<V.rows();i++){
    for(int j = 0;j<3;j++){
      U[i][j] = V(i,j);
    }
  }
  solid_angle.init(F.rows(), F.data(), V.rows(), &U[0], order);

  igl::parallel_for(T.rows(),[&](int q)
  //for(int q = 0;q<T.rows();q++)
  {
    HDK_Sample::UT_Vector3T<float>Qq;
    Qq[0] = T(q,0);
    Qq[1] = T(q,1);
    Qq[2] = T(q,2);
    Wapprox(q) = solid_angle.computeSolidAngle(Qq, accuracy_scale)/(4.0*M_PI);
  }
  ,1000);

  endTime = chrono::high_resolution_clock::now();
  auto wApproxTime = chrono::duration_cast<chrono::milliseconds>(endTime-startTime).count();
  
  cout<<"\n\nComputing fast approximation winding numbers finished in time "<<wApproxTime<<" ms"<<endl;

  // saving aprox winding numbers
  igl::writeDMAT(outApproxFile, Wapprox, true);

  /* =================================
       Visualize Winding Numbers
  ===================================*/

  // Plot the generated mesh
  igl::opengl::glfw::Viewer vWinding;
  update_visualization(vWinding);
  vWinding.callback_key_down = &key_down;
  vWinding.launch();
}
