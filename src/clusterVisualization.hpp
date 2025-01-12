#ifndef CLUSTER_VISUALIZATION_H
#define CLUSTER_VISUALIZATION_H

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkErrorCode.h>

#include "dataStructures.hpp"

int visualizePoints() {
    // Step 1: Create points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->InsertNextPoint(0.0, 0.0, 0.0);  // Point 1 (origin)
    points->InsertNextPoint(1.0, 1.0, 0.0);  // Point 2 (on the X-Y plane)
    points->InsertNextPoint(1.0, 0.0, 1.0);  // Point 3 (on the X-Z plane)

    // Step 2: Create PolyData object and set points
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);

    // Step 3: Use a glyph filter to represent each point as a vertex
    vtkSmartPointer<vtkVertexGlyphFilter> vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexGlyphFilter->SetInputData(polyData);
    vertexGlyphFilter->Update();

    // Step 4: Create a mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());

    // Step 5: Create an actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Step 6: Create a renderer and set background color
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(0.1, 0.1, 0.1);  // Dark background

    // Step 7: Create a render window (without the interactor)
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    // Ensure that render window has a valid size
    renderWindow->SetSize(800, 600);

    // Step 8: Set up a camera for the scene (ensure the points are visible)
    vtkSmartPointer<vtkCamera> camera = renderer->GetActiveCamera();
    camera->SetPosition(5.0, 5.0, 5.0); // Set camera position
    camera->SetFocalPoint(0.0, 0.0, 0.0); // Look at the origin
    camera->SetViewUp(0.0, 1.0, 0.0); // Set the view-up direction
    renderer->ResetCamera(); // Reset the camera to view the points

    // Print diagnostics to check render window status
    std::cout << "Render window size: " << renderWindow->GetSize()[0] << "x" << renderWindow->GetSize()[1] << std::endl;

    // Step 9: Use a window-to-image filter to capture the rendered image
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    
    // Check if the filter is properly set
    if (!windowToImageFilter->GetInput()) {
        std::cerr << "Error: vtkWindowToImageFilter is not correctly connected to the render window." << std::endl;
        return EXIT_FAILURE;
    }

    try {
        windowToImageFilter->Update();  // This is the line causing the crash
    } catch (const std::exception &e) {
        std::cerr << "Exception during Update(): " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Step 10: Write the captured image to a PNG file
    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName("output.png");
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    
    // Write and check for errors
    try {
        writer->Write();
    } catch (const std::exception &e) {
        std::cerr << "Exception during Write(): " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Rendered image saved to output.png" << std::endl;
    return EXIT_SUCCESS;
}

#endif // CLUSTER_VISUALIZATION_H