**Building**:
- Go to the project root directory.
- Run "source utility/load_modules.sh".
- Then make a directory for the build, go into it.
- Run "ccmake ..". Most dependencies should pick up automatically (you might need to set HDF5 by hand).
- Configure and generate. Ignore the warnings.
- Run "make".

**Preparation**:
- Add an HDF5 attribute called "VectorSpacing" to the Vervet1818.h5 dataset. This attribute is a 3-value array which tells the distance in the XY-plane and the distance in the Z-direction:
![alt text](/docs/images/Tutorial1.png)
- Note that the MSA0309.h5 has an attribute like this already, called "Voxelsize" as seen below:
![alt text](/docs/images/Tutorial2.png)

**Running**:
- Copy "utility/run.sh" to the directory where the executable is and run it:
![alt text](/docs/images/Tutorial3.png)
- If you want to display Vervet1818.h5-like data (where each slice is a separate dataset) toggle "Slice by slice". If you want to display MSA0309.h5-like data (where all slices are stored in a single volume, indexed as ZVXY) toggle "Volume".
- Click browse and select the dataset.
- Click the "Selector" plugin:
![alt text](/docs/images/Tutorial4.png)
- Select a region of the dataset and click "Update Viewer". The retardation and fiber orientation maps will appear:
![alt text](/docs/images/Tutorial5.png)
- For ODFs: Ensure the selection is a power of two (e.g. 2048x2048x1) and click the "Fiber Distribution Maps" plugin, adjust the parameters as you like, and click "Calculate".
![alt text](/docs/images/Tutorial6.png)

**Examples**:
- The following are example visualizations extracted from the application.
![alt text](/docs/images/Example1.png)
![alt text](/docs/images/Example2.png)
![alt text](/docs/images/Example3.png)
![alt text](/docs/images/Example4.png)
![alt text](/docs/images/Example5.png)

**Limitations**:
- For the hierarchical ODF tree to work correctly, one must use power-of-2 selections.