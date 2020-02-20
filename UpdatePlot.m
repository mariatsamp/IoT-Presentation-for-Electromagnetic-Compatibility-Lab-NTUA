function UpdatePlot(h,model,tour)

    if ~isempty(tour)
        tour=[tour tour(1)];
    end
    x2=model.x(tour);
    y2=model.y(tour);
    set(h{1},'XDataSource','x2');
    set(h{1},'YDataSource','y2');
    refreshdata(h(1),'caller');
    
end