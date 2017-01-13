package visualization;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Font;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;

import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.ui.swingViewer.View;
import org.graphstream.ui.swingViewer.Viewer;

import graph.ComputeGraph;

public class VisualizeGraph
{
	
	public VisualizeGraph(ComputeGraph graph)
	{
		System.setProperty("org.graphstream.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");
		String styleSheet=null;
		try 
		{
			styleSheet = new String(Files.readAllBytes(new File(System.getProperty("user.dir")+"/src/visualization/userTransfersGraphStyleSheet.css").toPath()));
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		graph.addAttribute("ui.stylesheet", styleSheet);
		
		for(Node node: graph.getNodeSet())
		{
			node.addAttribute("ui.label", node.getId());
		}
		
		JFrame frame = new JFrame();
        frame.setSize(320, 240);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        graph.addAttribute("ui.quality");
        graph.addAttribute("ui.antialias");
        
        Viewer viewer=new Viewer(graph, Viewer.ThreadingModel.GRAPH_IN_SWING_THREAD);
        viewer.enableAutoLayout();
        View view=viewer.addDefaultView(false);
        //new UIUpdater(view, graph).start();
        
        frame.setLayout(new BorderLayout());
        frame.addMouseWheelListener(new GraphMouseWheelListener(view));
        new MousePositionListener(view, (Component)view).start();
        frame.add((Component)view, BorderLayout.CENTER);
        
        JPanel listPane = new JPanel();
        listPane.setLayout(new BorderLayout());
        
        frame.add(listPane, BorderLayout.EAST);
        frame.setVisible(true);
	}

}

class UIUpdater extends Thread
{
	
	View view;
	Graph graph;
	
	UIUpdater(View view, Graph graph)
	{
		this.view=view;
		this.graph=graph;
	}
	
	public void run() 
	{
        while(true)
        {
        	double zoomLevel=view.getCamera().getViewPercent();
        	for(Node node: graph.getEachNode())
        	{
        		if((double)node.getAttribute("willie_visLevel")<zoomLevel)
        		{
        			node.addAttribute("ui.hide");
        		}
        		else
        		{
        			node.removeAttribute("ui.hide");
        		}
        	}
        }
	}
	
}

class MousePositionListener extends Thread
{
	
	double moveBounds=0.5;
	double moveSpeed=1.0;
	double moveGradient=1.75;
	View view;
	Component frame;
	
	MousePositionListener(View view, Component frame)
	{
		this.view=view;
		this.frame=frame;
	}
	
	public void run() 
	{
        while(true)
        {
        	Point mousePoint=MouseInfo.getPointerInfo().getLocation();
        	SwingUtilities.convertPointFromScreen(mousePoint, frame);
        	Rectangle graphFrame=frame.getBounds();
        	
        	int minX=(int)Math.min(mousePoint.getX()-graphFrame.getMinX(), graphFrame.getMaxX()-mousePoint.getX());
        	int minY=(int)Math.min(mousePoint.getY()-graphFrame.getMinY(), graphFrame.getMaxY()-mousePoint.getY());
        	
        	if(minX>=0 && minY>=0
        			&& (minX<moveBounds*graphFrame.getWidth() || minY<moveBounds*graphFrame.getHeight()))
        	{
        		double diag=Math.sqrt(Math.pow(graphFrame.getWidth(), 2)+Math.pow(graphFrame.getHeight(), 2));
        		double pxToGU=view.getCamera().getGraphDimension()/diag;
        		double deltaX=view.getCamera().getViewPercent()*pxToGU*moveSpeed*Math.signum(mousePoint.getX()-graphFrame.getCenterX())*Math.pow(Math.max(Math.abs(mousePoint.getX()-graphFrame.getCenterX())-Math.abs((1.0-moveBounds)*graphFrame.getCenterX()), 0.0), moveGradient)/graphFrame.getWidth();
        		double deltaY=view.getCamera().getViewPercent()*pxToGU*moveSpeed*Math.signum(graphFrame.getCenterY()-mousePoint.getY())*Math.pow(Math.max(Math.abs(graphFrame.getCenterY()-mousePoint.getY())-Math.abs((1.0-moveBounds)*graphFrame.getCenterY()), 0.0), moveGradient)/graphFrame.getHeight();
        		
        		view.getCamera().setViewCenter(view.getCamera().getViewCenter().x+deltaX,
        				view.getCamera().getViewCenter().y+deltaY,
        				view.getCamera().getViewCenter().z);
        	}
        	try 
        	{
				Thread.sleep(5);
			} 
        	catch (InterruptedException e) 
        	{
				e.printStackTrace();
			}
        }
    }
	
}

class GraphMouseWheelListener implements MouseWheelListener
{
	
	View view;
	double viewPercent=1.0;
	double viewScale=0.9;
	
	public GraphMouseWheelListener(View view)
	{
		this.view=view;
	}

	@Override
	public void mouseWheelMoved(MouseWheelEvent e)
	{
		if(e.getWheelRotation()<0)
		{
			viewPercent*=Math.pow(viewScale, -e.getWheelRotation());
			view.getCamera().setViewPercent(viewPercent);
		}
		else if(e.getWheelRotation()>0)
		{
			viewPercent/=Math.pow(viewScale, e.getWheelRotation());
			view.getCamera().setViewPercent(viewPercent);
		}
		
	}
	
}

class GraphMouseListener implements MouseListener
{
	
	View view;

	public GraphMouseListener(View view)
	{
		this.view=view;
	}
	
	@Override
	public void mouseClicked(MouseEvent e) 
	{
		// TODO Auto-generated method stub
		
	}

	double moveFactor=1.5;
	@Override
	public void mousePressed(MouseEvent e) 
	{
		int x=e.getX();
		int y=e.getY();
		int curX=(int)view.getCamera().getViewCenter().x;
		int curY=(int)view.getCamera().getViewCenter().y;
		
		int newX=(int)(curX+(x-curX)*moveFactor);
		int newY=(int)(curY+(y-curY)*moveFactor);
		
		view.getCamera().setViewCenter(newX, newY, view.getCamera().getViewCenter().z);
	}

	@Override
	public void mouseReleased(MouseEvent e) 
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseEntered(MouseEvent e) 
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseExited(MouseEvent e) 
	{
		// TODO Auto-generated method stub
		
	}
	
}
