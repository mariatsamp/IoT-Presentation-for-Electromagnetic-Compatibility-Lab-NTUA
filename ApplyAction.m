function tour2=ApplyAction(tour1,action)

    i=action(1);
    j=action(2);
    a=action(3);
    
    tour2=tour1;
    
    switch a
        case 1
            % Swap
            tour2([i j])=tour2([j i]);
            
        case 2
            % Reversion
            if i<j
                tour2(i:j)=tour2(j:-1:i);
            else
                tour2(j:i)=tour2(i:-1:j);
            end
            
        case 3
            % Insertion
            if i<j
                tour2=[tour2(1:i) tour2(j) tour2(i+1:j-1) tour2(j+1:end)];
            else
                tour2=[tour2(1:j-1) tour2(j+1:i) tour2(j) tour2(i+1:end)];
            end
            
    end

end