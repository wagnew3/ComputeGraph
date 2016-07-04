package visualization;

import org.graphstream.graph.Graph;
import org.graphstream.ui.swingViewer.Viewer;
import org.graphstream.ui.swingViewer.ViewerListener;
import org.graphstream.ui.swingViewer.ViewerPipe;

public class ClickChanger extends Thread implements ViewerListener
{
	
	ClickChange mainClass;
	ViewerPipe fromViewer;
	Viewer viewer;
	Graph graph;
	
	public ClickChanger(ClickChange mainClass, Viewer viewer, Graph graph)
	{
		this.mainClass=mainClass;
		this.viewer=viewer;
		this.graph=graph;
		fromViewer=viewer.newViewerPipe();
		fromViewer.addViewerListener(this);
		fromViewer.addSink(graph);
		start();
	}
	
	public void run()
	{
		while(true)
		{
			try
			{
			fromViewer.pump();
			}
			catch(Exception e)
			{
				e.printStackTrace();
				fromViewer.clearSinks();
				fromViewer=viewer.newViewerPipe();
				fromViewer.addViewerListener(this);
				fromViewer.addSink(graph);
			}
			try 
			{
				Thread.sleep(0);
			} 
			catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
		}
	}

	@Override
	public void viewClosed(String viewName) 
	{
		
	}

	@Override
	public void buttonPushed(String id) 
	{
		mainClass.updateClicked(id);
	}

	@Override
	public void buttonReleased(String id) 
	{
		mainClass.updateReleased(id);
	}

}
