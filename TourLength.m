function L=TourLength(tour,D)

    n=numel(tour);
    tour=[tour tour(1)];
    L=0;
    for i=1:n
        L=L+D(tour(i),tour(i+1));
    end
    
end