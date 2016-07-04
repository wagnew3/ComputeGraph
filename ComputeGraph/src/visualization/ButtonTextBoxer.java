package visualization;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JTextArea;

public class ButtonTextBoxer implements ActionListener
{
	
	JButton button;
	JTextArea textBox;
	ButtonTextBox mainClass;
	
	public ButtonTextBoxer(JButton button, JTextArea textBox, ButtonTextBox mainClass)
	{
		this.button=button;
		this.textBox=textBox;
		this.mainClass=mainClass;
		button.addActionListener(this);
	}

	@Override
	public void actionPerformed(ActionEvent e) 
	{
		String str=textBox.getText();
		mainClass.buttonPressed(str);
	}

}
