classdef RegressionL2Loss < dagnn.Loss
  properties
    crop = [0,0,0,0];
  end

  methods
    function outputs = forward(obj, inputs, params)%inputs H*W*C*B
      outputs{1} = L2FocalLoss(inputs{1}, inputs{2}(obj.crop(1)+1:end-obj.crop(2), obj.crop(3)+1:end-obj.crop(4),:,:));
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs{1} = L2FocalLoss(inputs{1}, inputs{2}(obj.crop(1)+1:end-obj.crop(2), obj.crop(3)+1:end-obj.crop(4),:,:), derOutputs{1}) ;
        derInputs{2} = [] ;      
        derParams = {} ;
    end

    function obj = RegressionL2Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
