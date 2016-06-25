/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.model.naivebayes;


/**
 *
 * @author daniele
 */
public class NaiveBayes {

    public static void main(String[] args) {
        
        Double accuracy;
        Classifier c = new Classifier();
        if(c.train()){
            accuracy = c.evalutate();
            System.out.println("Accuracy = " + accuracy);
        }else{
            System.out.println("Something wrong...");
        }
    }
    
}
